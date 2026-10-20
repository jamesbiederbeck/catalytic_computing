"""Proof-aware leases (RFC-003 A3): a lease is accepted only for a program
whose restoration of the borrowed registers is proven on its allocated
machine code, never because the program (or its job) says so.

`certify(image, window, gains, temps)` runs the clean-computation proof
(`isa_clean.verify_clean`, polynomial symbolic execution) on the image with
the window registers R24 .. R24+window-1 as the catalytic set whose entry
values the lease supplies. It fails closed on everything the proof cannot
follow (LD/ST, HORNER/REDP, FTOX, WAITI/RETI, a branch on a field value),
and also rejects:

  - a window register outside the lease, or a host input into the window
    (the library's R24..R31, or a private value there)
  - a borrowed register that does not come back as itself, an output that
    does not gain exactly the claimed polynomial, any other register that
    changes without being a declared temporary
  - a read of an entry value neither the host nor the lease supplies
  - an interrupt handler that names a window register, or does not pass
    `verify_handler` (the clean proof covers interrupt-free runs; a verified
    handler that stays off the window composes with it)

The result is a `Certificate` bound to the program's crc32 and the window
size. `check_lease` then binds it to an actual lease: the same program,
the same number of registers, an owner that is not the borrower, distinct
registers, and an owner whose interrupt handler (if it has one) touches
none of them. The register mapping itself is a renaming of the window: the
proof is about logical R24.., and the physical registers behind them are
the owner's, distinct, not shared with the borrower's own, and nobody else
writes them while the lease is held (exclusive lease, owner stall).

Domain: the proof holds for window values in [0, p) (`requires_canonical`
lists them); acquisition refuses a lease whose registers hold anything else,
in the model (`MultiCore.acquire`) and in the hardware (`K` replies 0xFE).

`Checker` is what the scheduler calls (`MultiCore(accept=...)`): it proves
each (program, window, claim) once and caches the certificate by crc.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from ..isa.encoding import decode
from ..isa.image import ProgramImage
from ..isa.lease import WINDOW, Lease, field_operands, window_conflicts
from .isa_catalytic import IsaCatalyticError, referenced_registers, verify_handler
from .isa_clean import CleanProofError, Poly, verify_clean


class LeaseRefused(Exception):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("lease refused:\n  " + "\n  ".join(violations))


@dataclass(frozen=True)
class Claim:
    """What a borrowed computation is supposed to do: the polynomial each
    output register gains, and the private registers it may leave changed.
    The proof checks the code against this; nothing here is trusted."""
    gains: tuple[tuple[int, str], ...]
    temps: tuple[int, ...] = ()
    polys: tuple = field(default=(), compare=False, hash=False)

    @classmethod
    def of(cls, gains: dict[int, Poly], temps=()) -> Claim:
        g = tuple(sorted((r, str(p)) for r, p in gains.items()))
        return cls(g, tuple(sorted(temps)), tuple(sorted(gains.items(),
                                                         key=lambda x: x[0])))


@dataclass(frozen=True)
class Certificate:
    image_crc: int
    window: int
    catalytic: tuple[int, ...]
    outputs: tuple[tuple[int, str], ...]
    temps: tuple[int, ...]
    requires_canonical: tuple[int, ...]
    retired: int
    cycles: int
    handler: str

    @property
    def key(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()[:16]


def _handler_pcs(image: ProgramImage) -> range:
    code = [decode(w) for w in image.words]
    end = image.handler
    while end < len(code) and code[end].op != "RETI":
        end += 1
    return range(image.handler, min(end + 1, len(code)))


def handler_registers(image: ProgramImage) -> set[int]:
    """Field registers the interrupt handler names (empty if none)."""
    if image.handler is None:
        return set()
    out = set()
    for pc in _handler_pcs(image):
        out |= referenced_registers(decode(image.words[pc]))
    return out


def certify(image: ProgramImage, window: int, claim: Claim) -> Certificate:
    """Prove that `image`, given `window` leased registers at R24.., brings
    them back and does what `claim` says. Raises LeaseRefused."""
    errs = list(window_conflicts(image, window))
    win = [WINDOW + i for i in range(window)]
    handler = "none"
    if image.handler is not None:
        touch = sorted(handler_registers(image) & set(win))
        if touch:
            errs.append(f"the interrupt handler names window registers "
                        f"{['R%d' % r for r in touch]}")
        try:
            verify_handler(image)
            handler = "verified, off the window"
        except IsaCatalyticError as e:
            errs += [f"handler: {v}" for v in e.violations]
    gains = dict(claim.polys)
    try:
        rep = verify_clean(image, win, gains, list(claim.temps),
                           supplied=tuple(win))
    except CleanProofError as e:
        errs += e.violations
        rep = None
    if errs:
        raise LeaseRefused(errs)
    return Certificate(image.crc, window, tuple(win), claim.gains, claim.temps,
                       tuple(rep.requires_canonical), rep.retired, rep.cycles,
                       handler)


def check_lease(cert: Certificate, image: ProgramImage, lease: Lease,
                borrower: int, owner_image: Optional[ProgramImage] = None
                ) -> list[str]:
    """Does `cert` cover running `image` on lane `borrower` with `lease`?"""
    errs = []
    if cert.image_crc != image.crc:
        errs.append(f"certificate is for program {cert.image_crc:#010x}, "
                    f"not {image.crc:#010x}")
    if cert.window != len(lease.regs):
        errs.append(f"certificate covers {cert.window} window registers, "
                    f"the lease has {len(lease.regs)}")
    if lease.owner == borrower:
        errs.append("owner and borrower are the same lane: the window would "
                    "alias the borrower's own registers")
    if len(set(lease.regs)) != len(lease.regs):
        errs.append(f"lease maps a register twice: {lease.regs}")
    if owner_image is not None:
        touch = sorted(handler_registers(owner_image) & set(lease.regs))
        if touch:
            errs.append(f"the owner's interrupt handler names leased "
                        f"registers {['R%d' % r for r in touch]}")
    return errs


class Checker:
    """Lease acceptance for `MultiCore(accept=Checker())`. Each variant must
    carry a `Claim`; the checker proves it (once per program, window and
    claim) and binds the certificate to the lease."""

    def __init__(self):
        self.cache: dict[tuple, Certificate | LeaseRefused] = {}
        self.proofs = 0
        self.hits = 0
        self.refused: list[tuple[str, list[str]]] = []

    def certificate(self, image: ProgramImage, window: int,
                    claim: Claim) -> Certificate:
        key = (image.crc, window, claim)
        if key in self.cache:
            self.hits += 1
        else:
            self.proofs += 1
            try:
                self.cache[key] = certify(image, window, claim)
            except LeaseRefused as e:
                self.cache[key] = e
        got = self.cache[key]
        if isinstance(got, LeaseRefused):
            raise got
        return got

    def __call__(self, mc, lane: int, v, lease: Lease) -> list[str]:
        if not isinstance(v.claim, Claim):
            errs = ["no claim to prove: a lease is not granted on a "
                    "program's word"]
        else:
            try:
                cert = self.certificate(v.image, len(lease.regs), v.claim)
                ol = mc.lanes[lease.owner]
                errs = check_lease(cert, v.image, lease, lane,
                                   ol.variant.image if ol.variant else None)
                if not errs:
                    v.certificate = cert
            except LeaseRefused as e:
                errs = e.violations
        if errs:
            self.refused.append((v.image.listing[0] if v.image.listing else "",
                                 errs))
        return errs

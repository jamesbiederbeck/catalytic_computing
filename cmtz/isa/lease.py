"""Register leases for multilane cmtz-core (RFC-003).

A lease lends up to eight of one lane's physical field registers (the
owner's) to another lane (the borrower). The borrower reaches them through
its remote window: logical registers R24.. R24+n-1 address the leased
registers themselves, not copies. While the lease is held, the owner may run
but stalls on any instruction that touches a leased register.

The window sits in R24..R31, the range the compiler never allocates
(`isa_backend.MAIN_POOL` is R1..R23) and that the runtime library
(`lib/matrix.s`, matpow) owns. So a program that links the library cannot
borrow; `window_conflicts` reports that.

`relocate` rewrites a program's field-register numbers, which is how a
hand-written program is moved onto the window; the result is a new image and
any proof about it has to be redone on it (`analysis/lease_proof.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .encoding import ALL_OPS, FORMATS, NUM_R, encode
from .encoding import decode as _decode


def decode(word: int):
    return _decode(word, ext=True)
from .image import ProgramImage

WINDOW = 24             # first window register in the borrower
LEASE_MAX = 8           # registers per lease
BANK = 8                # registers per bank (static hardware leases)


@dataclass(frozen=True)
class Lease:
    """`regs[i]` is the owner's register behind the borrower's R(24 + i)."""
    owner: int
    regs: tuple[int, ...]

    def __post_init__(self):
        if not 0 < len(self.regs) <= LEASE_MAX:
            raise ValueError(f"a lease holds 1..{LEASE_MAX} registers")
        if len(set(self.regs)) != len(self.regs):
            raise ValueError(f"lease maps one register twice: {self.regs}")
        if any(not 0 < r < NUM_R for r in self.regs):
            raise ValueError(f"lease registers must be in R1..R31: {self.regs}")

    @classmethod
    def banked(cls, owner: int, bank: int, count: int) -> Lease:
        """Registers bank*8 .. bank*8 + count - 1 (bank 1..3): the form the
        hardware supports."""
        if not 1 <= bank <= 3:
            raise ValueError("bank must be 1..3 (bank 0 holds R0)")
        return cls(owner, tuple(bank * BANK + i for i in range(count)))

    @property
    def window(self) -> list[int]:
        return [WINDOW + i for i in range(len(self.regs))]

    @property
    def bank(self) -> tuple[int, int] | None:
        """(bank, count) if this lease has the banked form, else None."""
        b, k = divmod(self.regs[0], BANK)
        if k == 0 and self.regs == tuple(range(self.regs[0],
                                               self.regs[0] + len(self.regs))):
            return b, len(self.regs)
        return None


def field_operands(op: str) -> list[str]:
    """Instruction fields ('a', 'b', 'c') holding field-register numbers."""
    return [fld for kind, fld in FORMATS[ALL_OPS[op].fmt] if kind == "r"]


def relocate(image: ProgramImage, mapping: dict[int, int],
             drop_inputs: tuple[str, ...] = ()) -> ProgramImage:
    """A copy of `image` with field register r renamed mapping.get(r, r)
    everywhere: in the code, the host register maps, the handler's borrowed
    and scratch sets. Names in `drop_inputs` stop being host-written (a
    lease supplies them). The mapping must be one-to-one on the registers
    the program uses."""
    if 0 in mapping or 0 in mapping.values():
        raise ValueError("R0 cannot be relocated")
    used = set()
    for w in image.words:
        ins = decode(w)
        used |= {getattr(ins, f) for f in field_operands(ins.op)}
    used |= set(image.inputs.values()) | set(image.outputs.values())
    target = {r: mapping.get(r, r) for r in used - {0}}
    if len(set(target.values())) != len(target):
        raise ValueError(f"relocation is not one-to-one: {target}")
    words = []
    for w in image.words:
        ins = decode(w)
        ch = {f: mapping.get(getattr(ins, f), getattr(ins, f))
              for f in field_operands(ins.op)}
        words.append(encode(replace(ins, **ch)))
    m = lambda r: mapping.get(r, r)
    return replace(
        image, words=words,
        inputs={k: m(r) for k, r in image.inputs.items() if k not in drop_inputs},
        outputs={k: m(r) for k, r in image.outputs.items()},
        borrowed=sorted(m(r) for r in image.borrowed),
        handler_scratch=sorted(m(r) for r in image.handler_scratch),
        listing=[str(decode(w)) for w in words])


def referenced(image: ProgramImage) -> set[int]:
    """Every field register the code names (R0 excluded)."""
    out = set()
    for w in image.words:
        ins = decode(w)
        out |= {getattr(ins, f) for f in field_operands(ins.op)}
    return out - {0}


def window_conflicts(image: ProgramImage, count: int) -> list[str]:
    """Reasons `image` cannot run with a window of `count` registers: it
    names a window register outside the lease, or the host writes one."""
    win = set(range(WINDOW, WINDOW + count))
    errs = []
    for r in sorted(referenced(image) & set(range(WINDOW, NUM_R)) - win):
        errs.append(f"R{r} is in the window range but outside the lease "
                    f"(R{WINDOW}..R{WINDOW + count - 1}); the library or a "
                    "private value lives there")
    for k, r in image.inputs.items():
        if r in win:
            errs.append(f"input {k!r} is host-written into window register R{r}")
    return errs

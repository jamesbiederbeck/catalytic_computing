"""Jobs for the multilane experiments (RFC-003): the projection programs in
private and leased form, an owner program that keeps live values in
registers it does not touch for a long stretch, and random mixes.

Leased forms move a program's catalytic registers onto the window
(R24..), so the registers they occupy are the owner's, reached through a
lease; the host no longer writes them. `relocate` produces a new image,
and every proof is redone on that image (`analysis/lease_proof.py`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..projection_oracle import gate, gate_tree
from . import projection as PR
from .asm import assemble
from .image import ProgramImage
from .lease import WINDOW, Lease, relocate
from .multicore import Job, Variant

CORE = PR.CORE
P = CORE.p


@dataclass
class Leased:
    image: ProgramImage
    catalytic: dict[str, int]       # name -> window register
    private: PR.Demo                # the program it came from


def leased(demo: PR.Demo) -> Leased:
    """`demo` with its catalytic registers moved to R24, R25, ..."""
    cat = demo.layout.catalytic
    mapping = {r: WINDOW + i for i, r in enumerate(cat.values())}
    img = relocate(demo.image, mapping, drop_inputs=tuple(cat))
    return Leased(img, {k: mapping[r] for k, r in cat.items()}, demo)


# owner: x, c in R1, R2; live values h0..h2 in R16..R18, untouched until the
# loop is over; r = (loop result) + h0 + h1 + h2
OWNER_LIVE = (16, 17, 18)


def owner_program(n: int) -> ProgramImage:
    if not 0 < n < 2048:
        raise ValueError("n must be in 1..2047")
    src = f"""
        .entry main
main:   LIX   X1, 0
        LIX   X2, {n}
loop:   MUL   R3, R1, R2
        ADD   R1, R3, R2
        ADDIX X1, X1, 1
        BLTX  X1, X2, loop
        ADD   R4, R16, R17
        ADD   R4, R4, R18
        ADD   R5, R1, R4
        HALT
"""
    a = assemble(src)
    return ProgramImage(CORE.p, CORE.q, a.words, a.entry, a.handler,
                        inputs={"x": 1, "c": 2, "h0": 16, "h1": 17, "h2": 18},
                        outputs={"r": 5, "acc": 1},
                        listing=[str(i) for i in a.instrs])


def owner_expected(n: int, x: int, c: int, h: tuple[int, int, int]) -> dict:
    for _ in range(n):
        x = (x * c + c) % P
    return {"r": (x + sum(h)) % P, "acc": x}


def gate_job(name: str, tt: int, a: int, b: int, y: int, rng: random.Random,
             forms: str = "both", lane=None, arrival: int = 0) -> Job:
    """A gate_clean job. forms: 'private', 'leased' (window, for the dynamic
    policy) or 'both' (leased first)."""
    lg = leased(PR.gate_clean())
    ins = {**PR.gate_inputs(tt, a, b), "y": y,
           **{k: rng.randrange(P) for k in ("t1", "t2", "s")}}
    vs = []
    if forms in ("leased", "both"):
        vs.append(Variant(lg.image, "leased", window=len(lg.catalytic)))
    if forms in ("private", "both"):
        vs.append(Variant(PR.gate_clean().image, "private"))
    return Job(name, vs, ins, lane=lane, arrival=arrival,
               expect={"y": (y + gate(tt, a, b)) % P})


def tree_job(name: str, f: int, g1: int, g2: int, bits: tuple, y: int,
             rng: random.Random, forms: str = "both", lane=None) -> Job:
    lt = leased(PR.tree_clean())
    ins = {**PR.tree_inputs(f, g1, g2, *bits), "y": y,
           **{k: rng.randrange(P) for k in ("t1", "t2", "tl", "tr", "s")}}
    vs = []
    if forms in ("leased", "both"):
        vs.append(Variant(lt.image, "leased", window=len(lt.catalytic)))
    if forms in ("private", "both"):
        vs.append(Variant(PR.tree_clean().image, "private"))
    want = gate_tree(f, g1, g2, *bits, {k: ins[k] for k in
                                        ("y", "t1", "t2", "tl", "tr", "s")})
    return Job(name, vs, ins, lane=lane, expect={"y": want["y"]})


def static_gate(name: str, owner: int, regs: tuple[int, ...], tt: int, a: int,
                b: int, y: int, lane: int) -> Job:
    lg = leased(PR.gate_clean())
    return Job(name, [Variant(lg.image, "leased", lease=Lease(owner, regs))],
               {**PR.gate_inputs(tt, a, b), "y": y}, lane=lane,
               expect={"y": (y + gate(tt, a, b)) % P})


def owner_job(name: str, n: int, x: int, c: int, h: tuple[int, int, int],
              lane: int, after: tuple[str, ...] = ()) -> Job:
    return Job(name, [Variant(owner_program(n), "private")],
               {"x": x, "c": c, "h0": h[0], "h1": h[1], "h2": h[2]},
               lane=lane, after=after, expect=owner_expected(n, x, c, h))

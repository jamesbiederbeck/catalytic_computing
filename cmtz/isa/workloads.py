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
    claim: object = None            # lease_proof.Claim: y gains the gate


def leased(demo: PR.Demo) -> Leased:
    """`demo` with its catalytic registers moved to R24, R25, ..., and the
    claim a lease checker proves before lending it registers: each output
    gains ĝ (or the tree's f̂(ĝ1, ĝ2)), nothing else changes."""
    from ..analysis.lease_proof import Claim
    cat = demo.layout.catalytic
    mapping = {r: WINDOW + i for i, r in enumerate(cat.values())}
    img = relocate(demo.image, mapping, drop_inputs=tuple(cat))
    gain = PR.expected_gain(img.inputs)
    claim = Claim.of({img.outputs[k]: gain for k in demo.layout.outputs},
                     demo.layout.temps)
    return Leased(img, {k: mapping[r] for k, r in cat.items()}, demo, claim)


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
        vs.append(Variant(lg.image, "leased", window=len(lg.catalytic),
                          claim=lg.claim))
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
        vs.append(Variant(lt.image, "leased", window=len(lt.catalytic),
                          claim=lt.claim))
    if forms in ("private", "both"):
        vs.append(Variant(PR.tree_clean().image, "private"))
    want = gate_tree(f, g1, g2, *bits, {k: ins[k] for k in
                                        ("y", "t1", "t2", "tl", "tr", "s")})
    return Job(name, vs, ins, lane=lane, expect={"y": want["y"]})


def static_gate(name: str, owner: int, regs: tuple[int, ...], tt: int, a: int,
                b: int, y: int, lane: int) -> Job:
    lg = leased(PR.gate_clean())
    return Job(name, [Variant(lg.image, "leased", lease=Lease(owner, regs),
                              claim=lg.claim)],
               {**PR.gate_inputs(tt, a, b), "y": y}, lane=lane,
               expect={"y": (y + gate(tt, a, b)) % P})


def owner_job(name: str, n: int, x: int, c: int, h: tuple[int, int, int],
              lane: int, after: tuple[str, ...] = ()) -> Job:
    return Job(name, [Variant(owner_program(n), "private")],
               {"x": x, "c": c, "h0": h[0], "h1": h[1], "h2": h[2]},
               lane=lane, after=after, expect=owner_expected(n, x, c, h))


# ── dynamic leases (RFC-003 A4): the program takes its own lease ─────────────

def gate_dynamic(owner: int, bank: int, tail: int = 0,
                 restore: bool = True) -> ProgramImage:
    """gate_clean with its masks in owner's bank, taken by BORROW and given
    back by RELEASE; `tail` NOPs after RELEASE (the lease is already back
    while they run)."""
    from .asm import assemble
    loop = PR._clean_loop("loop", PR.A, PR.B, PR.K, PR.Y, WINDOW, WINDOW + 1,
                          WINDOW + 2, 1, 3)
    if not restore:
        loop = loop.replace(f"MULI  R{WINDOW}, R{WINDOW}, {PR.OMEGA}",
                            f"MULI  R{WINDOW}, R{WINDOW}, 9", 1)
    src = (f".entry main\nmain:   BORROW {owner}, {bank}, 3\n"
           f"        LIX   X3, {PR.M if restore else 15}{loop}\n"
           "        RELEASE\n" + "        NOP\n" * tail + "        HALT\n")
    a = assemble(src, ext=True)
    ins = {"a": PR.A, "b": PR.B, "k0": 3, "k1": 4, "k2": 5, "k3": 6, "y": PR.Y}
    return ProgramImage(CORE.p, CORE.q, a.words, a.entry, a.handler,
                        inputs=ins, outputs={"y": PR.Y},
                        listing=[str(i) for i in a.instrs])


def gate_dynamic_claim():
    from ..analysis.lease_proof import Claim
    ins = {"a": PR.A, "b": PR.B, "k0": 3, "k1": 4, "k2": 5, "k3": 6}
    return Claim.of({PR.Y: PR.expected_gain(ins)})


def dynamic_job(name: str, owner: int, bank: int, tt: int, a: int, b: int,
                y: int, lane: int, tail: int = 0, arrival: int = 0,
                restore: bool = True) -> Job:
    img = gate_dynamic(owner, bank, tail, restore)
    return Job(name, [Variant(img, "dynamic", claim=gate_dynamic_claim())],
               {**PR.gate_inputs(tt, a, b), "y": y}, lane=lane,
               arrival=arrival, expect={"y": (y + gate(tt, a, b)) % P})


# ── register-pressure variants (RFC-003 A5) ──────────────────────────────────
#
# The same computation in four forms, renumbered so that a program's
# private register footprint is its highest register (what a lane budget
# limits). Gate: inputs a, b, k0..k3 in R1..R6, y in R7. Tree: a..d, three
# sets of k in R1..R16, y in R17.
#
#   direct   one temporary (gate) or three (tree): 8 / 20 registers
#   clean    the projection, catalytic registers private: 10 / 22
#   leased   the projection, catalytic registers borrowed: 7 / 17 + 3 / 5
#   spill    direct, with inputs parked in data memory while their
#            registers serve as temporaries: 7 / 17 + 1 / 5 data words.
#            Not covered by the clean proof (LD/ST); checked against the
#            oracle and for restored inputs in the tests.

GATE_C = {"a": 1, "b": 2, "k0": 3, "k1": 4, "k2": 5, "k3": 6, "y": 7}
TREE_Y = 17


def _asm_image(src: str, inputs: dict, outputs: dict) -> ProgramImage:
    from .asm import assemble
    a = assemble(src)
    return ProgramImage(CORE.p, CORE.q, a.words, a.entry, a.handler,
                        inputs=dict(inputs), outputs=dict(outputs),
                        listing=[str(i) for i in a.instrs])


def gate_variants() -> dict[str, tuple[ProgramImage, int, int]]:
    """kind -> (image, window registers, spill words)."""
    lg = leased(PR.gate_clean())
    spill = _asm_image("""
        ST    R3, [X0+0]        ; park k0; R3 is the temporary
        MUL   R3, R6, R1
        ADD   R3, R3, R5
        MUL   R3, R3, R2
        MAC   R3, R4, R1
        ADD   R7, R7, R3
        LD    R3, [X0+0]        ; k0 back
        ADD   R7, R7, R3
        HALT
""", GATE_C, {"y": 7})
    return {
        "direct": (relocate(PR.gate_direct().image, {9: 7, 10: 8}), 0, 0),
        "clean": (relocate(PR.gate_clean().image,
                           {9: 7, 7: 8, 8: 9, 16: 10}), 0, 0),
        "leased": (relocate(lg.image, {9: 7}), 3, 0),
        "spill": (spill, 0, 1),
        # takes lane 0's R16..R18 itself with BORROW / RELEASE (A4)
        "borrow": (relocate(gate_dynamic(0, 2), {9: 7}), 0, 0),
    }


def tree_variants() -> dict[str, tuple[ProgramImage, int, int]]:
    lt = leased(PR.tree_clean())
    ins = {k: r for k, r in PR._tree_in().items()} | {"y": TREE_Y}
    spill = _asm_image("""
        ST    R1, [X0+0]        ; a
        ST    R5, [X0+1]        ; g1 k0; R5 becomes u
        MAC   R5, R6, R1
        MAC   R5, R7, R2
        MUL   R1, R1, R2        ; a b
        MAC   R5, R8, R1        ; u = g1(a, b)
        ST    R3, [X0+2]        ; c
        ST    R9, [X0+3]        ; g2 k0; R9 becomes v
        MAC   R9, R10, R3
        MAC   R9, R11, R4
        MUL   R3, R3, R4
        MAC   R9, R12, R3       ; v = g2(c, d)
        ST    R13, [X0+4]       ; f k0; R13 becomes f(u, v)
        MAC   R13, R14, R5
        MAC   R13, R15, R9
        MUL   R5, R5, R9
        MAC   R13, R16, R5
        ADD   R17, R17, R13
        LD    R1, [X0+0]
        LD    R5, [X0+1]
        LD    R3, [X0+2]
        LD    R9, [X0+3]
        LD    R13, [X0+4]
        HALT
""", ins, {"y": TREE_Y})
    return {
        "direct": (relocate(PR.tree_direct().image,
                            {21: 17, 17: 18, 18: 19, 22: 20}), 0, 0),
        "clean": (relocate(PR.tree_clean().image,
                           {21: 17, 17: 18, 18: 19, 19: 20, 20: 21}), 0, 0),
        "leased": (relocate(lt.image, {21: 17}), 5, 0),
        "spill": (spill, 0, 5),
    }


def footprint(image: ProgramImage, window: int) -> int:
    from .lease import referenced
    win = set(range(WINDOW, WINDOW + window))
    return max((referenced(image) | set(image.inputs.values())
                | set(image.outputs.values())) - win)


def pressure_job(name: str, kind: str, tt_or_tree, bits, y: int,
                 kinds: tuple[str, ...], lane=None, arrival: int = 0) -> Job:
    """A gate (tt_or_tree an int) or tree (a (f, g1, g2) triple) job offered
    in the listed forms; the claim for the leased form is proven before a
    lease (A3)."""
    from ..analysis.lease_proof import Claim
    tree = isinstance(tt_or_tree, tuple)
    vs = (tree_variants() if tree else gate_variants())
    if tree:
        f, g1, g2 = tt_or_tree
        ins = {**PR.tree_inputs(f, g1, g2, *bits), "y": y}
        want = (y + gate(f, gate(g1, bits[0], bits[1]),
                         gate(g2, bits[2], bits[3]))) % P
    else:
        ins = {**PR.gate_inputs(tt_or_tree, *bits), "y": y}
        want = (y + gate(tt_or_tree, *bits)) % P
    # the private clean form reads its masks from host-written registers
    mrng = random.Random(name)
    ins.update({k: mrng.randrange(P) for k in
                (("t1", "t2", "tl", "tr", "s") if tree else ("t1", "t2", "s"))})
    out = []
    for k in kinds:
        img, win, words = vs[k]
        claim = (Claim.of({img.outputs["y"]: PR.expected_gain(img.inputs)})
                 if win or k == "borrow" else None)
        out.append(Variant(img, k, window=win, claim=claim))
    return Job(name, out, ins, lane=lane, arrival=arrival,
               expect={"y": want})

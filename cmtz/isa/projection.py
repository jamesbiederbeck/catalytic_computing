"""Roots-of-unity projection on cmtz-core (F_17, ω = 3).

Hand-written programs for the experiment described in
`cmtz/projection_oracle.py`, which is the independent reference they are
tested against. Gates are runtime data: the four coefficients of the
multilinear extension ĝ(u, v) = k0 + k1·u + k2·v + k3·u·v sit in registers
the host writes, so one image covers all 16 two-input gates (and all 4096
gate triples of the tree).

  gate_direct      y += ĝ(a, b), straight-line; the baseline
  gate_projection  y -= Σ_j ĝ(a + ω^j τ1, b + ω^j τ2) with the masks
                   read-only and the shifted arguments built in temporaries
  gate_clean       the same sum computed in place: the masks are added
                   into, rotated and restored, and the product term goes
                   through a third restored register instead of a
                   temporary, so every register but y comes back
  gate_sweep       gate_clean for every pair of masks in F_17 inside one
                   run, results stored to data memory (exhaustive on the
                   board in one load per gate and input)
  gate_irq         gate_clean repeated, with an interrupt handler that
                   borrows a mask register in the middle of it
  tree_direct      y += f̂(ĝ1(a, b), ĝ2(c, d)), straight-line
  tree_clean       the same through two levels of projection; g1 and g2
                   only ever exist added into the parent's mask registers,
                   and every register but y comes back (only the loop
                   counters in X are overwritten)

Every register a program reads is host-written (`ProgramImage.inputs`).
`Layout` records what each register is for, so register counts in the
benchmarks come from the programs themselves.

Values: masks and accumulators must hold field values in [0, p). The
registers are rw = 5 bits wide and can hold up to q - 1 = 18; a mask that
starts at 17 or 18 is reduced by the first field operation and comes back
as 0 or 1. See `analysis/isa_clean.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..projection_oracle import GATES, coeffs
from .asm import assemble
from .image import CoreConfig, ProgramImage

CORE = CoreConfig(17, 19)
OMEGA = 3               # order 16 in F_17
M = 16                  # number of rotations; 16 = -1 mod 17
NEG = 16                # MULI by 16 negates


@dataclass
class Layout:
    """Register roles. `inputs` are read and never written; `catalytic`
    registers start with arbitrary field values and must come back;
    `outputs` gain the result; `temps` and `index` are free workspace the
    program overwrites."""
    inputs: dict[str, int]
    catalytic: dict[str, int] = field(default_factory=dict)
    outputs: dict[str, int] = field(default_factory=dict)
    temps: list[int] = field(default_factory=list)
    index: list[int] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"inputs": len(self.inputs), "catalytic": len(self.catalytic),
                "outputs": len(self.outputs), "temps": len(self.temps),
                "index": len(self.index)}


@dataclass
class Demo:
    image: ProgramImage
    layout: Layout
    source: str


def _ext(dst: int, u: int, v: int, k: tuple[int, int, int, int]) -> str:
    """dst = k0 + k1·u + (k3·u + k2)·v"""
    k0, k1, k2, k3 = k
    return f"""
        MUL   R{dst}, R{k3}, R{u}
        ADD   R{dst}, R{dst}, R{k2}
        MUL   R{dst}, R{dst}, R{v}
        MAC   R{dst}, R{k1}, R{u}
        ADD   R{dst}, R{dst}, R{k0}"""


def _acc_sub(acc: int, u: int, v: int, k: tuple[int, int, int, int],
             s: int) -> str:
    """acc -= ĝ(u, v) with no free temporary: s holds an arbitrary value σ
    and gets it back. The product k3·u·v goes through s (s = σ + k3·u, so
    acc picks up σ·v + k3·u·v), and the σ·v is cancelled after s is
    restored."""
    k0, k1, k2, k3 = k
    return f"""
        MULI  R{acc}, R{acc}, {NEG}       ; -acc
        ADD   R{acc}, R{acc}, R{k0}
        MAC   R{acc}, R{k1}, R{u}
        MAC   R{acc}, R{k2}, R{v}
        MAC   R{s}, R{k3}, R{u}         ; s = sigma + k3 u
        MAC   R{acc}, R{s}, R{v}         ; -acc + g(u, v) + sigma v
        MULI  R{s}, R{s}, {NEG}
        MAC   R{s}, R{k3}, R{u}
        MULI  R{s}, R{s}, {NEG}         ; s = sigma
        MULI  R{acc}, R{acc}, {NEG}
        MAC   R{acc}, R{s}, R{v}         ; acc - g(u, v)"""


def _clean_loop(label: str, x: int, z: int, k, acc: int, t1: int, t2: int,
                s: int, cnt: int, lim: int) -> str:
    """acc += ĝ(x, z) by 16 rotations of the masks t1, t2 (needs X{lim} =
    16). On exit t1, t2 and s hold what they held on entry."""
    return f"""
        LIX   X{cnt}, 0
{label}: ADD   R{t1}, R{t1}, R{x}        ; t1 = x + w^j T1
        ADD   R{t2}, R{t2}, R{z}{_acc_sub(acc, t1, t2, k, s)}
        SUB   R{t1}, R{t1}, R{x}        ; uncompute
        SUB   R{t2}, R{t2}, R{z}
        MULI  R{t1}, R{t1}, {OMEGA}         ; rotate
        MULI  R{t2}, R{t2}, {OMEGA}
        ADDIX X{cnt}, X{cnt}, 1
        BLTX  X{cnt}, X{lim}, {label}"""


def _image(src: str, layout: Layout, *, handler_regs=None) -> Demo:
    a = assemble(src)
    host = {**layout.inputs, **layout.catalytic, **layout.outputs}
    shown = {**layout.outputs, **layout.catalytic}
    borrowed, scratch = handler_regs or ([], [])
    img = ProgramImage(CORE.p, CORE.q, a.words, a.entry, a.handler,
                       inputs=host, outputs=shown, borrowed=borrowed,
                       handler_scratch=scratch,
                       listing=[str(i) for i in a.instrs])
    return Demo(img, layout, src)


# one-gate register map: a b | k0..k3 | t1 t2 | y | s | temps from R10
A, B, K, T1, T2, Y, S = 1, 2, (3, 4, 5, 6), 7, 8, 9, 16
_GATE_IN = {"a": A, "b": B, "k0": 3, "k1": 4, "k2": 5, "k3": 6}


def gate_name(tt: int) -> str:
    return {v: k for k, v in GATES.items()}[tt]


def gate_inputs(tt: int, a: int, b: int, prefix: str = "") -> dict[str, int]:
    """Host values for gate truth table `tt` (bit 2a + b) on inputs a, b."""
    ks = dict(zip(("k0", "k1", "k2", "k3"), coeffs(tt, CORE.p)))
    return {"a": a, "b": b, **{prefix + n: v for n, v in ks.items()}}


def gate_direct() -> Demo:
    t = 10
    src = f"""
        .entry main
main:   {_ext(t, A, B, K).strip()}
        ADD   R{Y}, R{Y}, R{t}
        HALT
"""
    return _image(src, Layout(_GATE_IN, outputs={"y": Y}, temps=[t]))


def gate_projection() -> Demo:
    w, u, v, t = 10, 11, 12, 13
    src = f"""
        .entry main
main:   LIX   X3, {M}
        LIX   X1, 0
        LDI   R{w}, 1                 ; w = omega^j
loop:   MUL   R{u}, R{w}, R{T1}
        ADD   R{u}, R{u}, R{A}         ; u = a + w T1
        MUL   R{v}, R{w}, R{T2}
        ADD   R{v}, R{v}, R{B}         ; v = b + w T2{_ext(t, u, v, K)}
        SUB   R{Y}, R{Y}, R{t}
        MULI  R{w}, R{w}, {OMEGA}
        ADDIX X1, X1, 1
        BLTX  X1, X3, loop
        HALT
"""
    lay = Layout({**_GATE_IN, "t1": T1, "t2": T2}, outputs={"y": Y},
                 temps=[w, u, v, t], index=[1, 3])
    return _image(src, lay)


def gate_clean() -> Demo:
    src = f"""
        .entry main
main:   LIX   X3, {M}{_clean_loop("loop", A, B, K, Y, T1, T2, S, 1, 3)}
        HALT
"""
    lay = Layout(_GATE_IN, catalytic={"t1": T1, "t2": T2, "s": S},
                 outputs={"y": Y}, index=[1, 3])
    return _image(src, lay)


def gate_sweep() -> Demo:
    """For T1, T2 over F_17 (T1 outer): y = y0, run gate_clean, store
    (y, t1, t2) at D[3i], D[3i+1], D[3i+2], i = 17·T1 + T2."""
    y0, one = 11, 12
    src = f"""
        .entry main
main:   LIX   X3, {M}
        LIX   X6, {CORE.p}
        LIX   X7, 0
        LDI   R{one}, 1
        LDI   R{T1}, 0
        LIX   X4, 0
li:     LDI   R{T2}, 0
        LIX   X5, 0
lk:     MOV   R{Y}, R{y0}{_clean_loop("loop", A, B, K, Y, T1, T2, S, 1, 3)}
        ST    R{Y}, [X7]
        ST    R{T1}, [X7+1]
        ST    R{T2}, [X7+2]
        ADDIX X7, X7, 3
        ADD   R{T2}, R{T2}, R{one}
        ADDIX X5, X5, 1
        BLTX  X5, X6, lk
        ADD   R{T1}, R{T1}, R{one}
        ADDIX X4, X4, 1
        BLTX  X4, X6, li
        HALT
"""
    lay = Layout({**_GATE_IN, "y0": y0}, catalytic={"s": S},
                 temps=[one, T1, T2, Y],
                 index=[1, 3, 4, 5, 6, 7])
    return _image(src, lay)


def sweep_expected(tt: int, a: int, b: int, y0: int) -> list[int]:
    from ..projection_oracle import masked_gate
    out = []
    for t1 in range(CORE.p):
        for t2 in range(CORE.p):
            out += masked_gate(tt, a, b, t1, t2, y0)
    return out


ISR_K, ISR_OUT = 14, 15


def gate_irq(outer: int, inner: int, restore: bool = True) -> Demo:
    """gate_clean repeated outer x inner times (y gains that many g); the
    handler borrows mask t1 and uses it, then restores it (or not)."""
    if not (0 < outer < 2048 and 0 < inner < 2048):
        raise ValueError("loop counts must be in 1..2047")
    src = f"""
        .entry main
        .handler isr
main:   LIX   X3, {M}
        LIX   X4, 0
        LIX   X6, {outer}
        LIX   X7, {inner}
ro:     LIX   X5, 0
ri:     {_clean_loop("loop", A, B, K, Y, T1, T2, S, 1, 3).strip()}
        ADDIX X5, X5, 1
        BLTX  X5, X7, ri
        ADDIX X4, X4, 1
        BLTX  X4, X6, ro
        HALT
isr:    LDI   R{ISR_K}, 4
        ADD   R{T1}, R{T1}, R{ISR_K}     ; t1 + 4
        MULI  R{T1}, R{T1}, 3
        MUL   R{ISR_OUT}, R{T1}, R{T1}   ; use it
        MULI  R{T1}, R{T1}, 6            ; 18 (t1 + 4) = t1 + 4
        {"SUB   R%d, R%d, R%d" % (T1, T1, ISR_K) if restore else "NOP"}
        RETI
"""
    lay = Layout(_GATE_IN, catalytic={"t1": T1, "t2": T2, "s": S},
                 outputs={"y": Y}, index=[1, 3, 4, 5, 6, 7])
    return _image(src, lay, handler_regs=([T1], [ISR_K, ISR_OUT]))


# tree register map: a b c d | g1 k | g2 k | f k | t1 t2 tl tr | y | tmp
TA, TB, TC, TD = 1, 2, 3, 4
KG1, KG2, KF = (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)
TT1, TT2, TL, TR, TY, TS = 17, 18, 19, 20, 21, 22


def _tree_in() -> dict[str, int]:
    d = {"a": TA, "b": TB, "c": TC, "d": TD}
    for name, ks in (("g1", KG1), ("g2", KG2), ("f", KF)):
        d.update({f"{name}_k{i}": r for i, r in enumerate(ks)})
    return d


def tree_inputs(f: int, g1: int, g2: int, a: int, b: int, c: int, d: int
                ) -> dict[str, int]:
    vals = {"a": a, "b": b, "c": c, "d": d}
    for name, tt in (("g1", g1), ("g2", g2), ("f", f)):
        vals.update({f"{name}_k{i}": k for i, k in enumerate(coeffs(tt, CORE.p))})
    return vals


def tree_direct() -> Demo:
    u, v, t = 17, 18, 22
    src = f"""
        .entry main
main:   {_ext(u, TA, TB, KG1).strip()}{_ext(v, TC, TD, KG2)}{_ext(t, u, v, KF)}
        ADD   R{TY}, R{TY}, R{t}
        HALT
"""
    return _image(src, Layout(_tree_in(), outputs={"y": TY}, temps=[u, v, t]))


def tree_clean() -> Demo:
    """Parent loop over j (X1): tl += g1, tr += g2 (child loops, X2),
    y -= f(tl, tr), tl -= g1, tr -= g2, rotate tl and tr. Both children use
    the masks t1, t2."""
    src = f"""
        .entry main
g1:     {_clean_loop("g1l", TA, TB, KG1, TL, TT1, TT2, TS, 2, 3).strip()}
        RET
g2:     {_clean_loop("g2l", TC, TD, KG2, TR, TT1, TT2, TS, 2, 3).strip()}
        RET
main:   LIX   X3, {M}
        LIX   X1, 0
pl:     CALL  g1                        ; tl = w^j TL + g1
        CALL  g2                        ; tr = w^j TR + g2{_acc_sub(TY, TL, TR, KF, TS)}
        MULI  R{TL}, R{TL}, {NEG}
        CALL  g1
        MULI  R{TL}, R{TL}, {NEG}         ; tl -= g1
        MULI  R{TR}, R{TR}, {NEG}
        CALL  g2
        MULI  R{TR}, R{TR}, {NEG}         ; tr -= g2
        MULI  R{TL}, R{TL}, {OMEGA}
        MULI  R{TR}, R{TR}, {OMEGA}
        ADDIX X1, X1, 1
        BLTX  X1, X3, pl
        HALT
"""
    lay = Layout(_tree_in(),
                 catalytic={"t1": TT1, "t2": TT2, "tl": TL, "tr": TR, "s": TS},
                 outputs={"y": TY}, index=[1, 2, 3])
    return _image(src, lay)


def rtl_cases(seed: int = 0) -> list:
    """`CoreCase`s for the RTL testbench: every program, interrupts at
    random boundaries, clock cycles checked against `Machine.cycles`."""
    import random

    from ..backends.isa_core import CoreCase

    rng = random.Random(seed)
    p = CORE.p

    def masks(*names: str) -> dict[str, int]:
        return {k: rng.randrange(p) for k in names}

    cases = []
    for fn in (gate_direct, gate_projection, gate_clean):
        demo = fn()
        for tt in (GATES["AND"], GATES["XOR"], GATES["NAND"], rng.randrange(16)):
            a, b = rng.randrange(2), rng.randrange(2)
            vals = {**gate_inputs(tt, a, b), **masks("y")}
            vals.update({k: v for k, v in masks("t1", "t2", "s").items()
                         if k in demo.image.inputs})
            cases.append(CoreCase(demo.image, f"{fn.__name__} {tt} {a}{b}",
                                  vals, check_cycles=True))
    cases.append(CoreCase(gate_sweep().image, "gate_sweep XOR 1 0",
                          {**gate_inputs(GATES["XOR"], 1, 0), "y0": 7, "s": 11},
                          check_cycles=True))
    for fn in (tree_direct, tree_clean):
        for _ in range(2):
            tts = [rng.randrange(16) for _ in range(3)]
            ins = [rng.randrange(2) for _ in range(4)]
            vals = {**tree_inputs(*tts, *ins), **masks("y")}
            if fn is tree_clean:
                vals.update(masks("t1", "t2", "tl", "tr", "s"))
            cases.append(CoreCase(fn().image, f"{fn.__name__} {tts} {ins}",
                                  vals, check_cycles=True))
    for restore in (True, False):
        img = gate_irq(2, 1, restore).image
        for k in (0, 1, 50, 237, 400):
            vals = {**gate_inputs(rng.randrange(16), 1, 1), **masks("t1", "t2", "s", "y")}
            cases.append(CoreCase(img, f"gate_irq restore={restore} irq@{k}",
                                  vals, irq_at=(k,), check_cycles=True))
    return cases


def prove(demo: Demo):
    """Run the clean-computation proof (`analysis/isa_clean.py`) on `demo`:
    the output gains ĝ (or f̂(ĝ1, ĝ2)) as a polynomial in the input and
    coefficient registers, for every value in F_17 of every register."""
    from ..analysis.isa_clean import verify_clean

    lay = demo.layout
    return verify_clean(demo.image, list(lay.catalytic.values()),
                        {r: expected_gain(lay.inputs) for r in lay.outputs.values()},
                        lay.temps)


def expected_gain(ins: dict[str, int]):
    """ĝ(a, b), or f̂(ĝ1(a, b), ĝ2(c, d)) for a tree, as a polynomial in the
    registers `ins` names (the input map of the program's image)."""
    from ..analysis.isa_clean import Poly

    p = CORE.p

    def ext(ks, u: Poly, v: Poly) -> Poly:
        k0, k1, k2, k3 = (Poly.var(p, ins[k]) for k in ks)
        return k0 + k1 * u + k2 * v + k3 * u * v

    var = lambda name: Poly.var(p, ins[name])
    if "g1_k0" in ins:
        kk = lambda g: tuple(f"{g}_k{i}" for i in range(4))
        return ext(kk("f"), ext(kk("g1"), var("a"), var("b")),
                   ext(kk("g2"), var("c"), var("d")))
    return ext(("k0", "k1", "k2", "k3"), var("a"), var("b"))


BENCH = [gate_direct, gate_projection, gate_clean, tree_direct, tree_clean]
CLOCK_HZ = 27_000_000       # Tang Nano 20K crystal; the core runs on it


def benchmark() -> list[dict]:
    """Per program: size, instructions and cycles (exact for every input:
    the clean-computation proof follows the one path all inputs take, and
    it is checked against the simulator), and registers by role."""
    rows = []
    for fn in BENCH:
        demo = fn()
        rep = prove(demo)
        lay = demo.layout
        rows.append({
            "program": fn.__name__, "words": len(demo.image.words),
            "instructions": rep.retired, "cycles": rep.cycles,
            "us_at_27MHz": round(rep.cycles / CLOCK_HZ * 1e6, 2),
            **lay.counts(),
            "field_regs": sum(len(v) for v in (lay.inputs, lay.catalytic,
                                                lay.outputs, lay.temps)),
            "not_restored": len(lay.temps) + len(lay.outputs),
        })
    return rows

"""Hand-written cmtz-core programs that the compiler cannot produce yet.

`catalytic_loop` is the interrupt demonstration. Main runs a nested loop
that reads a borrowed register on every iteration; the handler changes that
register (add a temporary, rotate, use it, rotate back, subtract) and
restores it. Wherever the interrupt lands, main's output must equal the
uninterrupted run. With `restore=False` the handler leaves the register
changed: `isa_catalytic.verify_handler` rejects it, and main's output
changes for an interrupt that lands mid-loop.

`alu_sweep` applies one field instruction to every operand pair and stores
the results in data memory, so a single run checks that instruction
exhaustively (in simulation and on the board).
"""

from __future__ import annotations

from .asm import assemble
from .image import CoreConfig, ProgramImage


def catalytic_loop(outer: int = 6, inner: int = 5,
                   restore: bool = True) -> ProgramImage:
    """F_17 program; input `x` (borrowed, R2), output `acc` (R3)."""
    if not (0 < outer < 2048 and 0 < inner < 2048):
        raise ValueError("loop counts must be in 1..2047")
    src = f"""
        .entry main
        .handler isr
main:   LDI   R3, 0             ; acc
        LDI   R4, 1             ; w
        LIX   X1, 0
        LIX   X3, {outer}
        LIX   X4, {inner}
outer:  LIX   X2, 0
inner:  MAC   R3, R2, R4        ; acc += x * w   (reads the borrowed x)
        MULI  R4, R4, 3         ; w *= 3, a generator of F_17*
        ADDIX X2, X2, 1
        BLTX  X2, X4, inner
        ADD   R3, R3, R2        ; acc += x
        ADDIX X1, X1, 1
        BLTX  X1, X3, outer
        HALT
isr:    LDI   R5, 4             ; temporary t
        ADD   R2, R2, R5        ; x + t
        MULI  R2, R2, 3         ; 3(x + t)
        MUL   R6, R2, R2        ; scratch use of the borrowed value
        MULI  R2, R2, 6         ; 18(x + t) = x + t
        {"SUB   R2, R2, R5" if restore else "NOP"}
        RETI
"""
    a = assemble(src)
    return ProgramImage(17, 19, a.words, a.entry, a.handler,
                        inputs={"x": 2}, outputs={"acc": 3, "isr_sq": 6},
                        borrowed=[2], handler_scratch=[5, 6],
                        listing=[str(i) for i in a.instrs])


# op -> (instruction text with R1 = a, R2 = b, result in R3; a range; b range)
SWEEPS = {
    "ADD":    ("ADD R3, R1, R2", "p", "p"),
    "SUB":    ("SUB R3, R1, R2", "p", "p"),
    "MUL":    ("MUL R3, R1, R2", "p", "p"),
    "MAC":    ("MOV R3, R1\n        MAC R3, R1, R2", "p", "p"),
    "HORNER": ("MOV R3, R1\n        HORNER R3, R2, {k}", "q", "p"),
    "REDP":   ("REDP R3, R1", "q", "1"),
}


def alu_sweep(core: CoreConfig, op: str, k: int = 0) -> ProgramImage:
    """Store `op` applied to every (a, b) at D[a * nb + b] (slot `out`).

    For MULI pass op="MULI" and the immediate as `k`; for HORNER, `k` is
    the added coefficient.
    """
    p, q = core.p, core.q
    if op == "MULI":
        body, ra, rb = f"MULI R3, R1, {k}", "p", "1"
    else:
        body, ra, rb = SWEEPS[op]
        body = body.format(k=k)
    na = p if ra == "p" else q
    nb = {"p": p, "q": q, "1": 1}[rb]
    if na * nb > core.dmem_size:
        raise ValueError("sweep does not fit in data memory")
    # a + 1: ADD while a < p - 1; past p, step mod q with HORNER (a*1 + 1)
    inc_a = ("ADD   R1, R1, R4" if ra == "p" else
             "HORNER R1, R4, 1")
    src = f"""
        .entry main
main:   LDI   R1, 0             ; a
        LDI   R4, 1             ; one
        LIX   X1, 0             ; address
        LIX   X5, 0             ; a counter
        LIX   X6, {na}
        LIX   X7, {nb}
loop_a: LDI   R2, 0             ; b
        LIX   X2, 0             ; b counter
loop_b: {body}
        ST    R3, [X1]
        ADDIX X1, X1, 1
        ADDIX X2, X2, 1
        BEQX  X2, X7, next_a
        ADD   R2, R2, R4        ; b < p - 1 here, so b + 1 stays < p
        JMP   loop_b
next_a: ADDIX X5, X5, 1
        BEQX  X5, X6, done
        {inc_a}
        JMP   loop_a
done:   HALT
"""
    a = assemble(src)
    return ProgramImage(p, q, a.words, a.entry,
                        listing=[str(i) for i in a.instrs])


def sweep_expected(core: CoreConfig, op: str, k: int = 0) -> list[int]:
    """What `alu_sweep` leaves in D[0 : na * nb]."""
    p, q = core.p, core.q
    ra, rb = ("p", "1") if op == "MULI" else SWEEPS[op][1:]
    na = p if ra == "p" else q
    nb = {"p": p, "q": q, "1": 1}[rb]
    f = {
        "ADD": lambda a, b: (a + b) % p,
        "SUB": lambda a, b: (a - b) % p,
        "MUL": lambda a, b: a * b % p,
        "MAC": lambda a, b: (a + a * b) % p,
        "HORNER": lambda a, b: (a * b + k) % q,
        "REDP": lambda a, b: a % p,
        "MULI": lambda a, b: a * k % p,
    }[op]
    return [f(a, b) for a in range(na) for b in range(nb)]


def noncanonical_handler(outer: int = 6, inner: int = 5) -> ProgramImage:
    """A handler the verifier accepted before it checked handler reads: it
    subtracts and adds back R7, an affine identity on borrowed R2. Main
    sets R7 = 18 (LDI allows immediates up to q - 1), and the core's SUB
    computes 0 - 18 as 12 instead of 16 (rb + p - rc wraps), so with
    x = 0 the handler leaves R2 = 13. `verify_handler` now rejects it."""
    src = f"""
        .entry main
        .handler isr
main:   LDI   R7, 18
        LIX   X1, 0
        LIX   X3, {outer}
        LIX   X4, {inner}
outer:  LIX   X2, 0
inner:  ADDIX X2, X2, 1
        BLTX  X2, X4, inner
        ADDIX X1, X1, 1
        BLTX  X1, X3, outer
        HALT
isr:    SUB   R2, R2, R7
        ADD   R2, R2, R7
        RETI
"""
    a = assemble(src)
    return ProgramImage(17, 19, a.words, a.entry, a.handler,
                        inputs={"x": 2}, outputs={"x": 2}, borrowed=[2],
                        listing=[str(i) for i in a.instrs])

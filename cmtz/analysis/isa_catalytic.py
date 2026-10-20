"""Catalytic restore verifier for cmtz-core programs (allocated code).

`catalytic_verify` checks an IRCatalyticRegion by value *names*. After
register allocation several names share one physical register, so that
check says nothing about the machine code. This pass checks the handler
the core actually runs, over physical registers:

  - The handler is straight-line from its entry to RETI: no branches,
    CALL, ST, WAITI or HALT, and no writes to index registers (v1).
  - It writes only its scratch pool and its borrowed registers.
  - Borrowed registers are written only with F_p-linear ops (ADD, SUB,
    MULI, MOV).
  - Each register's value is tracked as an affine form over the values
    *all* registers held at entry; anything nonlinear (MUL, MAC, HORNER,
    REDP, LD) is opaque. At RETI every borrowed register must be exactly
    its entry value. Since the form covers every register, a borrowed
    register left depending on a temporary or an input is caught too.

The interrupt can land between any two main instructions, so the entry
values are arbitrary; an identity form is what "restored" means.

The proof is over F_p, so it needs every borrowed register to hold a value
in [0, p) whenever the handler can run. Registers may also hold values in
[p, q) (HORNER accumulators, LDI immediates up to q - 1), so code outside
the handler must not write a borrowed register with HORNER or with an LDI
immediate >= p.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..isa.encoding import (
    BRANCH_OPS, LINEAR_OPS, NUM_R, decode, written_registers,
)
from ..isa.image import ProgramImage


class IsaCatalyticError(Exception):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("Catalytic restore invariant violated in the "
                         "interrupt handler:\n  " + "\n  ".join(violations))


# affine form over entry values: ({reg: coeff}, const), coefficients mod p;
# None = opaque (nonlinear)
_Form = Optional[tuple[dict[int, int], int]]


def _norm(coeffs: dict[int, int], const: int, p: int) -> _Form:
    return ({r: c % p for r, c in coeffs.items() if c % p}, const % p)


def _combine(x: _Form, y: _Form, sign: int, p: int) -> _Form:
    if x is None or y is None:
        return None
    c = dict(x[0])
    for r, k in y[0].items():
        c[r] = c.get(r, 0) + sign * k
    return _norm(c, x[1] + sign * y[1], p)


@dataclass
class HandlerReport:
    instructions: int
    borrowed: list[int]
    forms: dict[int, str]          # borrowed register -> final form, printed


def verify_handler(image: ProgramImage) -> Optional[HandlerReport]:
    """Verify `image`'s interrupt handler; None if it has none.

    Raises IsaCatalyticError listing every violation.
    """
    if image.handler is None:
        return None
    p = image.p
    borrowed = set(image.borrowed)
    allowed = borrowed | set(image.handler_scratch)
    errs: list[str] = []
    forms: list[_Form] = [({r: 1}, 0) if r else ({}, 0) for r in range(NUM_R)]

    # precondition: borrowed registers only ever hold values in [0, p)
    h_end = image.handler
    while h_end < len(image.words) and decode(image.words[h_end]).op != "RETI":
        h_end += 1
    for i, w in enumerate(image.words):
        if image.handler <= i <= h_end:
            continue
        ins = decode(w)
        wr = written_registers(ins)
        if wr and wr[0] == "R" and wr[1] in borrowed and (
                ins.op == "HORNER" or ins.op == "LDI" and ins.imm >= p):
            errs.append(f"pc {i} ({ins}): may leave borrowed R{wr[1]} "
                        f"outside [0, p), where the restore proof does not hold")

    pc, n = image.handler, 0
    while True:
        if pc >= len(image.words):
            errs.append("handler runs off the end of the program without RETI")
            break
        ins = decode(image.words[pc])
        where = f"pc {pc} ({ins})"
        n += 1
        if ins.op == "RETI":
            break
        if ins.op in BRANCH_OPS or ins.op in ("RET", "ST", "WAITI", "HALT"):
            errs.append(f"{where}: {ins.op} is not allowed in a handler (v1)")
            pc += 1
            continue
        w = written_registers(ins)
        if w and w[0] == "X":
            errs.append(f"{where}: writes an index register")
        elif w and w[1] != 0:
            r = w[1]
            if r not in allowed:
                errs.append(f"{where}: writes R{r}, which is neither borrowed "
                            "nor handler scratch")
            elif r in borrowed and ins.op not in LINEAR_OPS:
                errs.append(f"{where}: {ins.op} is not linear; borrowed R{r} "
                            "may only be changed with ADD, SUB, MULI or MOV")
            a, b, c = forms[ins.b], forms[ins.c], None
            if ins.op == "ADD":
                c = _combine(a, b, 1, p)
            elif ins.op == "SUB":
                c = _combine(a, b, -1, p)
            elif ins.op == "MULI" and a is not None:
                c = _norm({k: v * ins.imm for k, v in a[0].items()},
                          a[1] * ins.imm, p)
            elif ins.op == "MOV":
                c = a
            elif ins.op == "LDI":
                c = ({}, ins.imm % p) if ins.imm < p else None
            forms[r] = c
        pc += 1

    printed = {}
    for r in sorted(borrowed):
        f = forms[r]
        printed[r] = "opaque" if f is None else (
            " + ".join([f"{k}*R{s}" for s, k in sorted(f[0].items())]
                       + ([str(f[1])] if f[1] else [])) or "0")
        if f != ({r: 1}, 0):
            errs.append(f"R{r} is borrowed but leaves the handler as "
                        f"{printed[r]}, not R{r}")
    if errs:
        raise IsaCatalyticError(errs)
    return HandlerReport(n, sorted(borrowed), printed)

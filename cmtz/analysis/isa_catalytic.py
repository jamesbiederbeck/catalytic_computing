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

The proof is over F_p, but a register holds rw bits and can contain any
value below q (HORNER accumulators, LDI immediates up to q - 1, data memory
words, host writes). The ALU reduces such an operand, and its SUB wraps
for 0 - 18, so with a value in [p, q) an "identity" handler can corrupt a
borrowed register. So, for every register the handler reads before writing
it, and every borrowed register:

  - code outside the handler must never leave it in [p, q): no HORNER, LD
    or LDI >= p into it, and no MOV from a register that may hold such a
    value (`noncanonical_registers`, a fixpoint over the whole program);
  - a register the handler reads that is not borrowed must be one the host
    writes (`ProgramImage.inputs`), because a dispatch can come before
    main writes anything. (A borrowed register main has not written yet
    is a leftover main overwrites before reading it, so restoring it only
    mod p is harmless.)

`HandlerReport.requires_canonical` lists these registers: the host must
write values in [0, p) to the ones it writes (`CoreLink.run` checks).

Handler scratch registers must be disjoint from everything code outside
the handler references, or a dispatch would change a value main reads.
The allocator guarantees that for compiled code; this checks it for any
image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..isa.encoding import (
    BRANCH_OPS, FORMATS, LINEAR_OPS, NUM_R, OPS, Instr, decode,
    written_registers,
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
    requires_canonical: list[int]  # must hold [0, p) whenever a dispatch can come


def read_registers(ins: Instr) -> set[int]:
    """Field registers `ins` reads (R0 excluded)."""
    fmt = OPS[ins.op].fmt
    regs = {getattr(ins, fld) for kind, fld in FORMATS[fmt] if kind == "r"}
    w = written_registers(ins)
    if w and w[0] == "R" and ins.op not in ("MAC", "HORNER"):
        regs.discard(ins.a)             # a is only the destination
    if ins.op in ("LD", "LDI"):
        regs.clear()
    return regs - {0}


def referenced_registers(ins: Instr) -> set[int]:
    fmt = OPS[ins.op].fmt
    return {getattr(ins, fld) for kind, fld in FORMATS[fmt] if kind == "r"} - {0}


def noncanonical_registers(code: list[Instr], skip: range, p: int) -> set[int]:
    """Field registers that code outside `skip` may leave holding a value in
    [p, q): HORNER results, LDI >= p, LD (data memory holds up to q - 1),
    and MOV from such a register."""
    out: set[int] = set()
    changed = True
    while changed:
        changed = False
        for i, ins in enumerate(code):
            if i in skip:
                continue
            w = written_registers(ins)
            if not w or w[0] != "R" or w[1] == 0 or w[1] in out:
                continue
            if (ins.op in ("HORNER", "LD") or ins.op == "LDI" and ins.imm >= p
                    or ins.op == "MOV" and ins.b in out):
                out.add(w[1])
                changed = True
    return out


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

    code = [decode(w) for w in image.words]
    h_end = image.handler
    while h_end < len(code) and code[h_end].op != "RETI":
        h_end += 1
    handler_pcs = range(image.handler, h_end + 1)
    scratch = set(image.handler_scratch)
    for i, ins in enumerate(code):
        if i not in handler_pcs:
            for r in sorted(referenced_registers(ins) & scratch):
                errs.append(f"pc {i} ({ins}): uses R{r}, which is handler "
                            "scratch; a dispatch would change it under main")
    noncanon = noncanonical_registers(code, handler_pcs, p)
    reads: set[int] = set()
    written: set[int] = set()
    for i in handler_pcs:
        if i >= len(code):
            break
        reads |= read_registers(code[i]) - written
        w = written_registers(code[i])
        if w and w[0] == "R":
            written.add(w[1])
    host = set(image.inputs.values())
    for r in sorted(reads | borrowed):
        if r in noncanon:
            errs.append(f"R{r} is {'borrowed' if r in borrowed else 'read by the handler'}"
                        f" but code outside the handler may leave it in [p, q), "
                        "where the restore proof does not hold")
        elif r not in borrowed and r not in host:
            errs.append(f"the handler reads R{r}, which the host does not write: "
                        "a dispatch before main writes it sees a leftover, "
                        "possibly in [p, q)")

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
    return HandlerReport(n, sorted(borrowed), printed,
                         sorted((reads | borrowed) & host))

"""Per-instruction facts for a viewer: text, which registers it reads and writes,
where it branches. Everything comes from the operand formats in `encoding.py`."""

from __future__ import annotations

from .encoding import ALL_OPS, BRANCH_OPS, FORMATS, decode, format_instr, written_registers
from .image import ProgramImage

# ops that also read the register they write (Ra = f(Ra, ...))
_READS_DEST = {"MAC", "HORNER"}


def describe(pc: int, word: int) -> dict:
    """{pc, word, text, op, a, b, c, imm, reads, writes, mem, target} for one
    instruction (a, b, c, imm are the raw operand fields).

    `reads` and `writes` are register names ('R3', 'X1'); `mem` is 'read',
    'write' or None for the data memory; `target` is the branch or call
    destination (an absolute pc) or None."""
    ins = decode(word)
    fmt = ALL_OPS[ins.op].fmt
    vals = {"a": ins.a, "b": ins.b, "c": ins.c, "imm": ins.imm}
    ops = []                                   # (file, index) of each register operand, in order
    for kind, fld in FORMATS[fmt]:
        if kind in ("r", "x"):
            ops.append(("R" if kind == "r" else "X", vals[fld]))
        elif kind == "m":
            ops.append(("X", ins.b))           # the address register of [Xb + imm]
    dest = written_registers(ins)
    writes = [f"{dest[0]}{dest[1]}"] if dest and dest[1] != 0 else []
    reads = []
    for i, (file, idx) in enumerate(ops):
        if idx == 0:
            continue                           # R0 and X0 read as zero
        if dest and i == 0 and ins.op not in _READS_DEST:
            continue                           # the first operand is only written
        reads.append(f"{file}{idx}")
    if ins.op == "ST" and ("R" + str(ins.a)) not in reads and ins.a != 0:
        reads.insert(0, f"R{ins.a}")           # ST reads Ra, it writes memory
    seen: list[str] = []
    for r in reads:
        if r not in seen:
            seen.append(r)
    return {
        "pc": pc, "word": word, "op": ins.op, "text": format_instr(ins),
        "a": ins.a, "b": ins.b, "c": ins.c, "imm": ins.imm,
        "reads": seen, "writes": writes,
        "mem": "read" if ins.op == "LD" else "write" if ins.op == "ST" else None,
        "target": ins.imm if ins.op in BRANCH_OPS else None,
    }


def disassemble(image: ProgramImage) -> list[dict]:
    return [describe(pc, w) for pc, w in enumerate(image.words)]

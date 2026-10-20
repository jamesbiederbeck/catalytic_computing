"""Two-pass assembler for cmtz-core.

Syntax, one statement per line:

    ; comment
    label:                      ; labels may share a line with an instruction
        ADD   R1, R2, R3
        MULI  R1, R2, 15
        LD    R4, [X2+3]        ; or [X2]
        BLTX  X1, X7, label
    .entry   main               ; first instruction to run (default 0)
    .handler isr                ; interrupt handler entry (default none)

Several sources can be assembled into one program (`assemble(a, b, ...)`),
which is how compiled code and the runtime library in `lib/` are linked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .encoding import ALL_OPS, FORMATS, OPS, Instr, NUM_R, NUM_X, encode


class AsmError(ValueError):
    pass


@dataclass
class Assembled:
    instrs: list[Instr]
    labels: dict[str, int]
    entry: int = 0
    handler: int | None = None
    lines: list[str] = field(default_factory=list)    # source line per instr

    @property
    def words(self) -> list[int]:
        return [encode(i) for i in self.instrs]


_LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):")
_MEM = re.compile(r"^\[\s*X(\d+)\s*(?:\+\s*(\w+))?\s*\]$", re.I)


def _int(tok: str) -> int:
    try:
        return int(tok, 0)
    except ValueError:
        raise AsmError(f"expected a number, got {tok!r}") from None


def _reg(tok: str, prefix: str, limit: int) -> int:
    m = re.fullmatch(prefix + r"(\d+)", tok, re.I)
    if not m:
        raise AsmError(f"expected {prefix} register, got {tok!r}")
    n = int(m.group(1))
    if n >= limit:
        raise AsmError(f"{tok}: only {prefix}0..{prefix}{limit - 1} exist")
    return n


def _split_operands(text: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def assemble(*sources: str, ext: bool = False) -> Assembled:
    """`ext=True` also accepts the multilane extension (BORROW, RELEASE)."""
    known = ALL_OPS if ext else OPS
    stmts: list[tuple[str, list[str], str]] = []    # (op, operands, line)
    labels: dict[str, int] = {}
    directives: dict[str, str] = {}

    for src in sources:
        for lineno, raw in enumerate(src.splitlines(), 1):
            line = raw.split(";", 1)[0].strip()
            while (m := _LABEL.match(line)):
                name = m.group(1)
                if name in labels:
                    raise AsmError(f"line {lineno}: label {name!r} defined twice")
                labels[name] = len(stmts)
                line = line[m.end():].strip()
            if not line:
                continue
            head, _, rest = line.partition(" ")
            if head.startswith("."):
                if head not in (".entry", ".handler"):
                    raise AsmError(f"line {lineno}: unknown directive {head}")
                directives[head] = rest.strip()
                continue
            op = head.upper()
            if op not in known:
                raise AsmError(f"line {lineno}: unknown instruction {head!r}")
            stmts.append((op, _split_operands(rest), raw.strip()))

    def target(tok: str) -> int:
        if tok in labels:
            return labels[tok]
        return _int(tok)

    instrs: list[Instr] = []
    for op, ops, line in stmts:
        kinds = FORMATS[known[op].fmt]
        if len(ops) != len(kinds):
            raise AsmError(f"{line!r}: {op} takes {len(kinds)} operands")
        f = {"a": 0, "b": 0, "c": 0, "imm": 0}
        try:
            for tok, (kind, fld) in zip(ops, kinds):
                if kind == "r":
                    f[fld] = _reg(tok, "R", NUM_R)
                elif kind == "x":
                    f[fld] = _reg(tok, "X", NUM_X)
                elif kind in ("i", "s"):
                    f[fld] = _int(tok)
                elif kind == "l":
                    f[fld] = target(tok)
                elif kind == "m":
                    m = _MEM.match(tok)
                    if not m:
                        raise AsmError(f"expected [Xn+k], got {tok!r}")
                    f["b"] = _reg("X" + m.group(1), "X", NUM_X)
                    f["imm"] = _int(m.group(2)) if m.group(2) else 0
            ins = Instr(op, **f)
            encode(ins)     # range checks
        except (AsmError, ValueError) as e:
            raise AsmError(f"{line!r}: {e}") from None
        instrs.append(ins)

    entry = target(directives[".entry"]) if ".entry" in directives else 0
    handler = (target(directives[".handler"])
               if ".handler" in directives else None)
    return Assembled(instrs, labels, entry, handler,
                     [line for _, _, line in stmts])

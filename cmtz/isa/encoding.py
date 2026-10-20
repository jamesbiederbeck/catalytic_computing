"""cmtz-core instruction set: opcodes, operand formats, encode/decode.

This table is the single definition of the ISA. The assembler, the
simulator and the RTL generator (`backends/isa_core.py`) all read it, so a
change here changes every one of them together.

Word layout (32 bits):

    [31:26] op   [25:21] a   [20:16] b   [15:11] c   [10:0] imm

Register files:
    R0..R31   field registers, values in [0, p) (HORNER results in [0, q)).
              R0 reads as 0 and ignores writes.
    X0..X15   16-bit index registers for counters and addresses (plain
              binary, wrap on overflow). X0 reads as 0 and ignores writes.

Field operations require their operands in [0, p); the hardware reduces
with a single conditional subtract or a short subtract chain, so an
out-of-range operand gives a wrong (but deterministic) answer. The
simulator raises instead. Only HORNER and REDP see values in [0, q).
"""

from __future__ import annotations

from dataclasses import dataclass

ISA_VERSION = 1

NUM_R = 32
NUM_X = 16
X_BITS = 16
IMM_BITS = 11
STACK_DEPTH = 8


@dataclass(frozen=True)
class OpSpec:
    name: str
    code: int
    fmt: str        # operand format, see FORMATS
    doc: str


# fmt -> operand kinds in source order. 'r' field register, 'x' index
# register, 'i' unsigned immediate, 's' signed immediate, 'l' code label
# (absolute address in imm), 'm' memory operand [xb + imm].
# Each kind is placed in the field named alongside it.
FORMATS: dict[str, tuple[tuple[str, str], ...]] = {
    "N":   (),
    "RRR": (("r", "a"), ("r", "b"), ("r", "c")),
    "RR":  (("r", "a"), ("r", "b")),
    "RRI": (("r", "a"), ("r", "b"), ("i", "imm")),
    "RI":  (("r", "a"), ("i", "imm")),
    "XI":  (("x", "a"), ("i", "imm")),
    "XXS": (("x", "a"), ("x", "b"), ("s", "imm")),
    "XXX": (("x", "a"), ("x", "b"), ("x", "c")),
    "XR":  (("x", "a"), ("r", "b")),
    "RM":  (("r", "a"), ("m", "b")),
    "L":   (("l", "imm"),),
    "XXL": (("x", "a"), ("x", "b"), ("l", "imm")),
    "RL":  (("r", "a"), ("l", "imm")),
}

_OPS = [
    OpSpec("NOP",    0, "N",   "no operation"),
    OpSpec("HALT",   1, "N",   "stop; status = halted"),
    OpSpec("ADD",    2, "RRR", "Ra = (Rb + Rc) mod p"),
    OpSpec("SUB",    3, "RRR", "Ra = (Rb - Rc) mod p"),
    OpSpec("MUL",    4, "RRR", "Ra = (Rb * Rc) mod p"),
    OpSpec("MAC",    5, "RRR", "Ra = (Ra + Rb * Rc) mod p"),
    OpSpec("MULI",   6, "RRI", "Ra = (Rb * imm) mod p; imm < p (rotation by ω^j)"),
    OpSpec("MOV",    7, "RR",  "Ra = Rb"),
    OpSpec("LDI",    8, "RI",  "Ra = imm; imm < q"),
    OpSpec("HORNER", 9, "RRI", "Ra = (Ra * Rb + imm) mod q; imm < q"),
    OpSpec("REDP",  10, "RR",  "Ra = Rb mod p; Rb < q"),
    OpSpec("LIX",   11, "XI",  "Xa = imm"),
    OpSpec("ADDIX", 12, "XXS", "Xa = Xb + simm"),
    OpSpec("ADDX",  13, "XXX", "Xa = Xb + Xc"),
    OpSpec("MULX",  14, "XXX", "Xa = Xb * Xc (low 16 bits)"),
    OpSpec("FTOX",  15, "XR",  "Xa = Rb"),
    OpSpec("LD",    16, "RM",  "Ra = D[Xb + imm]"),
    OpSpec("ST",    17, "RM",  "D[Xb + imm] = Ra"),
    OpSpec("JMP",   18, "L",   "pc = imm"),
    OpSpec("BEQX",  19, "XXL", "if Xa == Xb: pc = imm"),
    OpSpec("BNEX",  20, "XXL", "if Xa != Xb: pc = imm"),
    OpSpec("BLTX",  21, "XXL", "if Xa < Xb (unsigned): pc = imm"),
    OpSpec("BZ",    22, "RL",  "if Ra == 0: pc = imm"),
    OpSpec("BNZ",   23, "RL",  "if Ra != 0: pc = imm"),
    OpSpec("CALL",  24, "L",   "push pc + 1; pc = imm"),
    OpSpec("RET",   25, "N",   "pc = pop"),
    OpSpec("RETI",  26, "N",   "leave the interrupt handler; pc = saved pc"),
    OpSpec("WAITI", 27, "N",   "stall until a handler has returned since the run began"),
]

OPS: dict[str, OpSpec] = {o.name: o for o in _OPS}
BY_CODE: dict[int, OpSpec] = {o.code: o for o in _OPS}

# Field ops that are F_p-linear in their register operands: what a strict
# interrupt handler may use to modify a borrowed register.
LINEAR_OPS = frozenset({"ADD", "SUB", "MULI", "MOV"})
BRANCH_OPS = frozenset({"JMP", "BEQX", "BNEX", "BLTX", "BZ", "BNZ", "CALL"})


@dataclass(frozen=True)
class Instr:
    op: str
    a: int = 0
    b: int = 0
    c: int = 0
    imm: int = 0        # signed for XXS, otherwise unsigned

    @property
    def spec(self) -> OpSpec:
        return OPS[self.op]

    def __str__(self) -> str:
        return format_instr(self)


def encode(ins: Instr) -> int:
    spec = OPS[ins.op]
    for v, n in ((ins.a, "a"), (ins.b, "b"), (ins.c, "c")):
        if not 0 <= v < 32:
            raise ValueError(f"{ins.op}: field {n}={v} out of range")
    imm = ins.imm
    if spec.fmt == "XXS":
        lo, hi = -(1 << (IMM_BITS - 1)), (1 << (IMM_BITS - 1)) - 1
        if not lo <= imm <= hi:
            raise ValueError(f"{ins.op}: signed imm {imm} out of range")
        imm &= (1 << IMM_BITS) - 1
    elif not 0 <= imm < (1 << IMM_BITS):
        raise ValueError(f"{ins.op}: imm {imm} out of range")
    return (spec.code << 26) | (ins.a << 21) | (ins.b << 16) | (ins.c << 11) | imm


def decode(word: int) -> Instr:
    code = (word >> 26) & 0x3F
    if code not in BY_CODE:
        raise ValueError(f"unknown opcode {code} in word {word:#010x}")
    spec = BY_CODE[code]
    imm = word & ((1 << IMM_BITS) - 1)
    if spec.fmt == "XXS" and imm >> (IMM_BITS - 1):
        imm -= 1 << IMM_BITS
    return Instr(spec.name, (word >> 21) & 31, (word >> 16) & 31,
                 (word >> 11) & 31, imm)


def format_instr(ins: Instr) -> str:
    spec = OPS[ins.op]
    vals = {"a": ins.a, "b": ins.b, "c": ins.c, "imm": ins.imm}
    parts = []
    for kind, fld in FORMATS[spec.fmt]:
        v = vals[fld]
        if kind == "r":
            parts.append(f"R{v}")
        elif kind == "x":
            parts.append(f"X{v}")
        elif kind == "m":
            parts.append(f"[X{ins.b}+{ins.imm}]")
        else:
            parts.append(str(v))
    return f"{ins.op} {', '.join(parts)}".rstrip()


def written_registers(ins: Instr) -> tuple[str, int] | None:
    """('R'|'X', index) of the register `ins` writes, or None."""
    fmt = OPS[ins.op].fmt
    if ins.op == "ST" or fmt in ("N", "L", "XXL", "RL"):
        return None
    kind = FORMATS[fmt][0][0]
    return ("R" if kind == "r" else "X", ins.a)

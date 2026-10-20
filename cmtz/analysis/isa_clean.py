"""Clean-computation proof for cmtz-core programs, on the machine code.

`isa_catalytic.verify_handler` tracks affine forms, so the first MUL makes
a register opaque. A Cook-Mertz projection restores its registers only
through a nonlinear cancellation (Σ_j ω^(jk) = 0), so proving it needs more:
here every field register holds a polynomial over F_p in the values all
registers held at entry (variable R<i>), and the program is executed
symbolically from its entry to HALT.

  - Index registers and control flow stay concrete: X registers must be
    set by LIX/ADDIX/ADDX/MULX from known values before a branch reads
    them, and BZ/BNZ may only test a register whose polynomial is a
    constant. So every run takes the same path, and the retired count and
    clock cycles come out exact for all inputs.
  - HORNER and REDP (mod q, outside F_p), LD and ST (data memory), FTOX,
    WAITI and RETI are not supported and stop the proof.
  - Polynomials are reduced with x^p = x, so two of them are equal exactly
    when they agree at every point of F_p^n: the comparisons below are
    statements about every possible register content in [0, p).

`verify_clean` then requires: catalytic registers come back as their own
variable, outputs come back as their variable plus the stated gain, and
every other register except declared temporaries is unchanged. That is the
proof for all entry values in [0, p), for the run without interrupts. An
interrupt handler that `verify_handler` accepts restores what it borrows
and writes only scratch registers main never touches, so a dispatch at any
boundary leaves main's run unchanged.

What it does not cover: register contents in [p, q). The register file is
rw bits wide and the host may write any value below q; a field operation
reduces such a value, so a mask that starts at 17 comes back as 0. The
report lists the registers whose entry values the proof depends on
(`requires_canonical`); the host must put values in [0, p) there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..isa.encoding import NUM_R, NUM_X, STACK_DEPTH, X_BITS, decode
from ..isa.image import ProgramImage

XMASK = (1 << X_BITS) - 1

Mono = tuple[tuple[int, int], ...]          # ((register, exponent), ...)


class Poly:
    """Polynomial over F_p in the entry values of R1..R31, reduced with
    x^p = x (so equality is equality of functions on F_p)."""

    __slots__ = ("p", "t")

    def __init__(self, p: int, terms: dict[Mono, int] | None = None):
        self.p = p
        self.t = {m: c % p for m, c in (terms or {}).items() if c % p}

    @classmethod
    def var(cls, p: int, r: int) -> Poly:
        return cls(p, {((r, 1),): 1}) if r else cls(p)

    @classmethod
    def const(cls, p: int, c: int) -> Poly:
        return cls(p, {(): c})

    def __add__(self, o: Poly) -> Poly:
        t = dict(self.t)
        for m, c in o.t.items():
            t[m] = t.get(m, 0) + c
        return Poly(self.p, t)

    def __neg__(self) -> Poly:
        return Poly(self.p, {m: -c for m, c in self.t.items()})

    def __sub__(self, o: Poly) -> Poly:
        return self + (-o)

    def scale(self, k: int) -> Poly:
        return Poly(self.p, {m: c * k for m, c in self.t.items()})

    def __mul__(self, o: Poly) -> Poly:
        p, t = self.p, {}
        for m1, c1 in self.t.items():
            for m2, c2 in o.t.items():
                e = dict(m1)
                for r, k in m2:
                    e[r] = e.get(r, 0) + k
                # x^p = x on F_p: fold exponents into 1..p-1
                m = tuple(sorted((r, (k - 1) % (p - 1) + 1) for r, k in e.items()))
                t[m] = t.get(m, 0) + c1 * c2
        return Poly(p, t)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Poly) and self.t == o.t

    def constant(self) -> int | None:
        if not self.t:
            return 0
        return self.t.get(()) if list(self.t) == [()] else None

    def evaluate(self, values: dict[int, int]) -> int:
        total = 0
        for m, c in self.t.items():
            for r, k in m:
                c *= pow(values[r], k, self.p)
            total += c
        return total % self.p

    def variables(self) -> set[int]:
        return {r for m in self.t for r, _ in m}

    def __str__(self) -> str:
        if not self.t:
            return "0"
        parts = []
        for m, c in sorted(self.t.items(), key=lambda mc: (len(mc[0]), mc[0])):
            mono = "*".join(f"R{r}" + (f"^{k}" if k > 1 else "") for r, k in m)
            parts.append(mono if c == 1 and mono else
                         f"{c}*{mono}" if mono else str(c))
        return " + ".join(parts)

    __repr__ = __str__


class CleanProofError(Exception):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("clean-computation proof failed:\n  "
                         + "\n  ".join(violations))


@dataclass
class SymRun:
    R: list[Poly]
    X: list[int | None]
    retired: int
    cycles: int
    # entry registers whose values reached a field operation
    reads: set[int] = field(default_factory=set)


def run_symbolic(image: ProgramImage, max_steps: int = 2_000_000) -> SymRun:
    """Execute `image` from its entry to HALT with symbolic field registers.
    Raises CleanProofError where the path or a value cannot be followed."""
    p = image.p
    code = [decode(w) for w in image.words]
    R = [Poly.var(p, r) for r in range(NUM_R)]
    X: list[int | None] = [0] + [None] * (NUM_X - 1)
    pc, stack, retired, cycles = image.entry, [], 0, 0
    reads: set[int] = set()

    def fail(msg: str):
        raise CleanProofError([f"pc {pc} ({code[pc]}): {msg}"])

    def xv(i: int) -> int:
        v = X[i]
        if v is None:
            fail(f"X{i} is not known here (set it with LIX first)")
        return v

    def setr(i: int, v: Poly) -> None:
        if i:
            R[i] = v

    def setx(i: int, v: int) -> None:
        if i:
            X[i] = v & XMASK

    for _ in range(max_steps):
        if not 0 <= pc < len(code):
            raise CleanProofError([f"pc {pc} outside the program"])
        ins = code[pc]
        op, a, b, c, imm = ins.op, ins.a, ins.b, ins.c, ins.imm
        nxt = pc + 1
        cycles += 2
        if op in ("ADD", "SUB", "MUL"):
            reads |= R[b].variables() | R[c].variables()
        elif op == "MAC":
            reads |= R[a].variables() | R[b].variables() | R[c].variables()
        elif op in ("MULI", "BZ", "BNZ"):
            reads |= R[b if op == "MULI" else a].variables()
        if op == "HALT":
            return SymRun(R, X, retired + 1, cycles, reads)
        if op == "NOP":
            pass
        elif op == "ADD":
            setr(a, R[b] + R[c])
        elif op == "SUB":
            setr(a, R[b] - R[c])
        elif op == "MUL":
            setr(a, R[b] * R[c])
        elif op == "MAC":
            setr(a, R[a] + R[b] * R[c])
        elif op == "MULI":
            setr(a, R[b].scale(imm))
        elif op == "MOV":
            setr(a, R[b])
        elif op == "LDI":
            if imm >= p:
                fail(f"LDI {imm} is outside [0, p)")
            setr(a, Poly.const(p, imm))
        elif op == "LIX":
            setx(a, imm)
        elif op == "ADDIX":
            setx(a, xv(b) + imm)
        elif op == "ADDX":
            setx(a, xv(b) + xv(c))
        elif op == "MULX":
            setx(a, xv(b) * xv(c))
        elif op == "JMP":
            nxt = imm
        elif op in ("BEQX", "BNEX", "BLTX"):
            x, y = xv(a), xv(b)
            if (op == "BEQX" and x == y or op == "BNEX" and x != y
                    or op == "BLTX" and x < y):
                nxt = imm
        elif op in ("BZ", "BNZ"):
            k = R[a].constant()
            if k is None:
                fail(f"branches on R{a} = {R[a]}, which depends on the inputs")
            if (k == 0) == (op == "BZ"):
                nxt = imm
        elif op == "CALL":
            if len(stack) >= STACK_DEPTH:
                fail("return stack overflow")
            stack.append(nxt)
            nxt = imm
        elif op == "RET":
            if not stack:
                fail("return stack underflow")
            nxt = stack.pop()
        else:
            fail(f"{op} is not supported by the clean-computation proof")
        pc = nxt
        retired += 1
    raise CleanProofError([f"no HALT within {max_steps} instructions"])


@dataclass
class CleanReport:
    retired: int
    cycles: int
    catalytic: list[int]
    outputs: dict[int, str]                 # register -> proven gain
    requires_canonical: list[int]           # entry values the ALU sees
    unchanged: list[int] = field(default_factory=list)


def verify_clean(image: ProgramImage, catalytic: list[int],
                 gains: dict[int, Poly], temps: list[int],
                 supplied: tuple[int, ...] = ()) -> CleanReport:
    """Prove that `image` adds `gains[r]` to each output register r, brings
    every catalytic register back, and changes nothing else except `temps`
    (and index registers), for every entry value in [0, p).

    Also checks that every entry value a field operation sees is in a
    host-written register (`image.inputs`) or in `supplied` (registers a
    lease provides, range-checked at acquisition): anything else is a
    leftover from an earlier program and may lie in [p, q), where the
    hardware's arithmetic is not arithmetic mod p (the proof would not
    apply even when the value cancels out). Raises CleanProofError listing
    every violation.
    """
    p = image.p
    run = run_symbolic(image)
    errs = []
    for r in catalytic:
        if run.R[r] != Poly.var(p, r):
            errs.append(f"catalytic R{r} ends as {run.R[r]}, not R{r}")
    for r, g in gains.items():
        got = run.R[r] - Poly.var(p, r)
        if got != g:
            errs.append(f"output R{r} gains {got}, expected {g}")
    free = set(catalytic) | set(gains) | set(temps) | {0}
    unchanged = []
    for r in range(1, NUM_R):
        if r in free:
            continue
        if run.R[r] != Poly.var(p, r):
            errs.append(f"R{r} is not declared a temporary but ends as {run.R[r]}")
        else:
            unchanged.append(r)
    used = set(run.reads) | set(catalytic)
    host = set(image.inputs.values()) | set(supplied)
    for r in sorted(used - host):
        errs.append(f"reads the entry value of R{r}, which the host does "
                    "not write: a leftover from an earlier program")
    if errs:
        raise CleanProofError(errs)
    return CleanReport(run.retired, run.cycles, sorted(catalytic),
                       {r: str(g) for r, g in gains.items()},
                       sorted(used), unchanged)

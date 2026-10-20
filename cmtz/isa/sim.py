"""Instruction-level simulator for cmtz-core.

This is the reference the RTL testbenches compare against: same register
files, data memory, return stack, status codes and retired-instruction
count. A malformed program (an operand outside the range its instruction
requires) is a compiler bug, not a machine state, so it raises `SimError`
instead of guessing what the hardware would do.

Run contract (what the core does on the host's `G`): `start()` resets the
control state only -- pc to the entry, return stack, pending/in-handler/
handled bits, status, error and the retired count. R, X and data memory
keep whatever they held (the host's `W`/`D` writes before `G`, and the
leftovers of earlier programs), so a program must not read a register or
data word it has not written and the host did not write. `scramble()`
fills those with random leftovers to check exactly that.

Interrupts: `irq_at` lists retired-instruction counts. At the instruction
boundary where `retired` equals one of them the pending bit is set. A
pending interrupt is dispatched at a boundary when the core is not already
in the handler: the pc is saved, the core jumps to the handler, and RETI
returns. A trigger during the handler stays pending until after RETI.
Without a handler, a trigger is dropped. WAITI stalls (without retiring)
until some handler has returned since the run began; a run that stalls
there with no trigger left to come returns with `waiting` set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .encoding import NUM_R, NUM_X, STACK_DEPTH, X_BITS, Instr, decode
from .image import CoreConfig, ProgramImage

# status codes, as the core reports them
IDLE, RUNNING, HALTED, ERROR = 0, 1, 2, 3
# error codes
E_NONE, E_OPCODE, E_PC, E_OVERFLOW, E_UNDERFLOW, E_RETI, E_ADDR, E_STEPS = range(8)
ERROR_NAMES = {E_NONE: "none", E_OPCODE: "bad opcode", E_PC: "pc out of range",
               E_OVERFLOW: "return stack overflow",
               E_UNDERFLOW: "return stack underflow",
               E_RETI: "RETI outside the handler",
               E_ADDR: "data address out of range", E_STEPS: "step limit"}

XMASK = (1 << X_BITS) - 1


class SimError(Exception):
    """The program broke an operand-range rule of the ISA."""


@dataclass
class Machine:
    core: CoreConfig
    image: ProgramImage
    R: list[int] = field(default_factory=lambda: [0] * NUM_R)
    X: list[int] = field(default_factory=lambda: [0] * NUM_X)
    D: list[int] = field(default_factory=list)
    pc: int = 0
    stack: list[int] = field(default_factory=list)
    status: int = IDLE
    error: int = E_NONE
    retired: int = 0
    in_handler: bool = False
    saved_pc: int = 0
    pending: bool = False
    handled: bool = False       # a RETI has happened since the run began
    waiting: bool = False       # stalled on WAITI
    # (retired count, saved pc) at each dispatch; the core keeps the last
    dispatches: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.image.validate(self.core)
        if not self.D:
            self.D = [0] * self.core.dmem_size
        self.code: list[Instr] = [decode(w) for w in self.image.words]
        self.pc = self.image.entry

    # ── host access ──────────────────────────────────────────────────

    def scramble(self, seed: int) -> None:
        """Fill R, X and data memory with random leftovers (R and data in
        [0, q), as a previous program could leave them)."""
        import random
        rng = random.Random(seed)
        q = self.core.q
        self.R = [0] + [rng.randrange(q) for _ in range(NUM_R - 1)]
        self.X = [0] + [rng.randrange(1 << X_BITS) for _ in range(NUM_X - 1)]
        self.D = [rng.randrange(q) for _ in range(self.core.dmem_size)]

    def set_inputs(self, values: dict[str, int]) -> None:
        """Inputs are field values, in [0, p)."""
        for name, v in values.items():
            if not 0 <= v < self.core.p:
                raise SimError(f"input {name!r} = {v} outside [0, p)")
            self.write_r(self.image.inputs[name], v)

    def write_r(self, r: int, v: int) -> None:
        if not 0 <= v < self.core.q:
            raise SimError(f"R{r} <- {v}: outside [0, q)")
        if r:
            self.R[r] = v

    def write_matrix(self, name: str, m: list[list[int]]) -> None:
        slot = self.image.matrices_in[name]
        if len(m) != slot.n or any(len(row) != slot.n for row in m):
            raise SimError(f"matrix {name!r} must be {slot.n}x{slot.n}")
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                if not 0 <= v < self.core.p:
                    raise SimError(f"matrix {name!r}[{i}][{j}]={v} outside [0, p)")
                self.D[slot.addr + i * slot.n + j] = v

    def read_matrix(self, name: str) -> list[list[int]]:
        slot = (self.image.matrices_out.get(name)
                or self.image.matrices_in[name])
        n = slot.n
        return [self.D[slot.addr + i * n: slot.addr + (i + 1) * n]
                for i in range(n)]

    def outputs(self) -> dict[str, int]:
        return {name: self.R[r] for name, r in self.image.outputs.items()}

    # ── execution ────────────────────────────────────────────────────

    def start(self) -> None:
        """Reset the control state for a run (the host's `G`)."""
        self.pc, self.stack = self.image.entry, []
        self.pending = self.in_handler = self.handled = self.waiting = False
        self.status, self.error, self.retired = RUNNING, E_NONE, 0
        self.saved_pc, self.dispatches = 0, []

    def run(self, max_steps: int = 1_000_000,
            irq_at: tuple[int, ...] | set[int] = ()) -> int:
        """Start a run and continue until HALT, an error, or a WAITI stall
        with no trigger left to come; returns the status."""
        irq = set(irq_at)
        self.start()
        for _ in range(max_steps):
            if self.retired in irq:
                irq.discard(self.retired)
                self.pending = True
            self.step()
            if self.status != RUNNING:
                return self.status
            if self.waiting and not self.pending and self.retired not in irq:
                return self.status
        self._fail(E_STEPS)
        return self.status

    def _fail(self, code: int) -> None:
        self.status, self.error = ERROR, code

    def _check(self, v: int, bound: int, what: str) -> int:
        if not 0 <= v < bound:
            raise SimError(f"pc {self.pc}: {what} = {v} outside [0, {bound})")
        return v

    def step(self) -> None:
        self.waiting = False
        if self.pending and not self.in_handler:
            self.pending = False
            if self.image.handler is not None:
                self.saved_pc, self.pc = self.pc, self.image.handler
                self.in_handler = True
                self.dispatches.append((self.retired, self.saved_pc))
        if not 0 <= self.pc < len(self.code):
            return self._fail(E_PC)
        ins = self.code[self.pc]
        p, q = self.core.p, self.core.q
        R, X = self.R, self.X
        nxt = self.pc + 1
        op, a, b, c, imm = ins.op, ins.a, ins.b, ins.c, ins.imm

        def rp(i: int) -> int:          # field operand, must be < p
            return self._check(R[i], p, f"R{i}")

        def setr(i: int, v: int) -> None:
            if i:
                R[i] = v

        def setx(i: int, v: int) -> None:
            if i:
                X[i] = v & XMASK

        if op == "NOP":
            pass
        elif op == "HALT":
            self.status = HALTED
            self.retired += 1
            return
        elif op == "ADD":
            setr(a, (rp(b) + rp(c)) % p)
        elif op == "SUB":
            setr(a, (rp(b) - rp(c)) % p)
        elif op == "MUL":
            setr(a, rp(b) * rp(c) % p)
        elif op == "MAC":
            setr(a, (rp(a) + rp(b) * rp(c)) % p)
        elif op == "MULI":
            setr(a, rp(b) * self._check(imm, p, "MULI imm") % p)
        elif op == "MOV":
            setr(a, R[b])
        elif op == "LDI":
            setr(a, self._check(imm, q, "LDI imm"))
        elif op == "HORNER":
            acc = self._check(R[a], q, f"R{a}")
            x = self._check(R[b], q, f"R{b}")
            setr(a, (acc * x + imm) % q)
        elif op == "REDP":
            setr(a, self._check(R[b], q, f"R{b}") % p)
        elif op == "LIX":
            setx(a, imm)
        elif op == "ADDIX":
            setx(a, X[b] + imm)
        elif op == "ADDX":
            setx(a, X[b] + X[c])
        elif op == "MULX":
            setx(a, X[b] * X[c])
        elif op == "FTOX":
            setx(a, R[b])
        elif op in ("LD", "ST"):
            addr = (X[b] + imm) & XMASK
            if addr >= self.core.dmem_size:
                return self._fail(E_ADDR)
            if op == "LD":
                setr(a, self.D[addr])
            else:
                self.D[addr] = R[a]
        elif op == "JMP":
            nxt = imm
        elif op in ("BEQX", "BNEX", "BLTX"):
            x, y = X[a], X[b]
            if (op == "BEQX" and x == y or op == "BNEX" and x != y
                    or op == "BLTX" and x < y):
                nxt = imm
        elif op in ("BZ", "BNZ"):
            if (R[a] == 0) == (op == "BZ"):
                nxt = imm
        elif op == "CALL":
            if len(self.stack) >= STACK_DEPTH:
                return self._fail(E_OVERFLOW)
            self.stack.append(nxt)
            nxt = imm
        elif op == "RET":
            if not self.stack:
                return self._fail(E_UNDERFLOW)
            nxt = self.stack.pop()
        elif op == "RETI":
            if not self.in_handler:
                return self._fail(E_RETI)
            self.in_handler = False
            self.handled = True
            nxt = self.saved_pc
        elif op == "WAITI":
            if not self.handled:
                self.waiting = True
                return              # stall: no retire, pc unchanged
        else:   # pragma: no cover - decode() rejects unknown opcodes
            return self._fail(E_OPCODE)
        self.pc = nxt
        self.retired += 1


def run_image(core: CoreConfig, image: ProgramImage,
              inputs: dict[str, int] | None = None,
              matrices: dict[str, list[list[int]]] | None = None,
              irq_at: tuple[int, ...] = (), max_steps: int = 1_000_000,
              scramble: int | None = None) -> Machine:
    """Load, run to completion and return the machine. `scramble` seeds
    random leftovers in R, X and data memory before the host writes."""
    m = Machine(core, image)
    if scramble is not None:
        m.scramble(scramble)
    m.set_inputs(inputs or {})
    for name, mat in (matrices or {}).items():
        m.write_matrix(name, mat)
    m.run(max_steps=max_steps, irq_at=irq_at)
    return m

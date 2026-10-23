"""A cmtz-core that lives in software and speaks the UART protocol of `host.py`.

`SimCore.feed(bytes)` takes what a host would send and returns what the board
would answer, so `CoreLink` and the browser visualizer can be exercised (and
tested) without hardware. It is a thin shell around `sim.Machine`, so what it
does to registers and memory is the simulator's by definition; what it adds is
the framing, the refusals and the debugger of the step build.

Differences from the board, all in timing:

* A free run (`G`) does not complete at once. Each command that arrives while
  the core is running first lets it execute `slice` instructions, so a host
  that polls `Q` sees it running, and `H` really stops it somewhere in the
  middle. The board runs the whole program in the time of a few bytes.
* An interrupt (`T`) is taken at the next step, like the simulator's `pending`.

The step build commands ('H', 'N', 'G' on a paused core, status 4) are always
present; `step=False` makes them unknown bytes, like a plain cmtz-core.
"""

from __future__ import annotations

from . import host
from .encoding import NUM_R, NUM_X
from .image import CoreConfig, ProgramImage
from .sim import IDLE, RUNNING, Machine, SimError

# bytes of arguments each command takes before it can run
_ARGS = {"P": 6, "W": 2, "D": 4, "M": 4}
_ONE = set("ISGQTX")


class SimCore:
    def __init__(self, core: CoreConfig, step: bool = True, slice: int = 64):
        self.core = core
        self.step = step
        self.slice = slice
        self.buf = bytearray()
        self.image: ProgramImage | None = None
        self.m: Machine | None = None
        self.paused = False
        # what survives between runs (registers, index registers, data memory,
        # the interrupt counter and the last dispatch), like the board's
        self.R = [0] * NUM_R
        self.X = [0] * NUM_X
        self.D = [0] * core.dmem_size
        self.irq_before = 0                       # dispatches in earlier runs
        self.last_disp = (0, 0)

    # ── plumbing ─────────────────────────────────────────────────────

    def feed(self, data: bytes) -> bytes:
        """Bytes in, the bytes the board would answer."""
        self.buf += data
        out = bytearray()
        while self.buf:
            cmd = chr(self.buf[0])
            known = _ONE | set(_ARGS) | (set("HN") if self.step else set())
            if cmd not in known:
                del self.buf[0]                   # ignored
                continue
            need = 1 + _ARGS.get(cmd, 0)
            if cmd == "P":
                if len(self.buf) < need:
                    break
                n = int.from_bytes(self.buf[1:3], "big")
                need += 4 * n
            elif cmd == "D":
                if len(self.buf) < need:
                    break
                need += int.from_bytes(self.buf[3:5], "big")
            if len(self.buf) < need:
                break
            frame = bytes(self.buf[:need])
            del self.buf[:need]
            self._advance()
            out += self._command(cmd, frame[1:])
        return bytes(out)

    def _advance(self) -> None:
        """Let a free-running core make progress between commands."""
        m = self.m
        if m is None or m.status != RUNNING or self.paused:
            return
        for _ in range(self.slice):
            if m.waiting and not m.pending:
                return
            self._one_step()
            if m.status != RUNNING:
                return

    def _one_step(self) -> None:
        """One `Machine.step()`. The simulator raises on an operand outside
        [0, p) where the chip computes a wrong answer; here that shows up as
        an error status (code 1), so a browser session does not crash."""
        try:
            self.m.step()
        except SimError:
            self.m._fail(1)
    @property
    def running(self) -> bool:
        return self.m is not None and self.m.status == RUNNING

    def _dispatches(self) -> list[tuple[int, int]]:
        return self.m.dispatches if self.m else []

    # ── commands ─────────────────────────────────────────────────────

    def _command(self, cmd: str, args: bytes) -> bytes:
        return getattr(self, "_c_" + cmd)(args)

    def _c_I(self, args: bytes) -> bytes:
        return host.info_frame(self.core, self.image)

    def _c_P(self, args: bytes) -> bytes:
        n = int.from_bytes(args[0:2], "big")
        entry = int.from_bytes(args[2:4], "big")
        handler = int.from_bytes(args[4:6], "big")
        raw = args[6:]
        bad = bytes([host.BAD] * 5)
        if (self.running or n == 0 or n > self.core.pmem_size or entry >= n
                or not (handler == 0xFFFF or handler < n)):
            return bad
        words = [int.from_bytes(raw[4 * i:4 * i + 4], "big") for i in range(n)]
        try:
            image = ProgramImage(self.core.p, self.core.q, words, entry=entry,
                                 handler=None if handler == 0xFFFF else handler)
            m = Machine(self.core, image)
        except Exception:
            return bad
        self._retire_run()
        self.image, self.m, self.paused = image, m, False
        m.R, m.X, m.D = self.R, self.X, self.D            # state is kept across loads
        m.status = IDLE
        return bytes([host.OK]) + host.words_crc(words).to_bytes(4, "big")

    def _c_W(self, args: bytes) -> bytes:
        r, v = args
        if self.running or r >= NUM_R or v >= self.core.q:
            return bytes([host.BAD])
        if r:
            self.R[r] = v
        return bytes([host.OK])

    def _c_D(self, args: bytes) -> bytes:
        addr = int.from_bytes(args[0:2], "big")
        count = int.from_bytes(args[2:4], "big")
        data = args[4:]
        if (self.running or addr + count > self.core.dmem_size
                or any(b >= self.core.q for b in data)):
            return bytes([host.BAD])
        self.D[addr:addr + count] = list(data)
        return bytes([host.OK])

    def _readable(self) -> bool:
        return not self.running or self.paused

    def _c_M(self, args: bytes) -> bytes:
        addr = int.from_bytes(args[0:2], "big")
        count = int.from_bytes(args[2:4], "big")
        if not self._readable() or addr + count > self.core.dmem_size:
            return bytes([host.BAD] * count)
        return bytes(self.D[addr:addr + count])

    def _c_S(self, args: bytes) -> bytes:
        if not self._readable():
            return bytes([host.BAD] * host.REGS_LEN)
        return bytes(self.R) + b"".join(x.to_bytes(2, "big") for x in self.X)

    def _c_G(self, args: bytes) -> bytes:
        if self.step and self.paused:
            self.paused = False
            return bytes([host.OK])
        if self.image is None or self.running:
            return bytes([host.BAD])
        self._begin()
        return bytes([host.OK])

    def _retire_run(self) -> None:
        """Fold the finished run's interrupt count and last dispatch into
        what the core keeps between runs."""
        if self.m is not None and self.m.dispatches:
            self.irq_before += len(self.m.dispatches)
            self.last_disp = self.m.dispatches[-1]

    def _begin(self) -> None:
        self._retire_run()
        self.m.start()

    def _c_Q(self, args: bytes) -> bytes:
        m = self.m
        if m is None:
            return host.Status(IDLE, 0, 0, self.irq_before, 0, 0, 0).frame()
        disp = m.dispatches[-1] if m.dispatches else self.last_disp
        status = host.PAUSED if self.paused else m.status
        return host.Status(status, m.error, m.retired,
                           self.irq_before + len(m.dispatches),
                           disp[0], disp[1], m.pc).frame()

    def _c_T(self, args: bytes) -> bytes:
        if self.running:
            self.m.pending = True
        return bytes([host.OK])

    def _c_X(self, args: bytes) -> bytes:
        if self.running:
            self.m._fail(7)
            self.paused = False
        return bytes([host.OK])

    # step build

    def _c_H(self, args: bytes) -> bytes:
        if self.image is None:
            return bytes([host.BAD])
        if not self.running:
            self._begin()                         # idle, halted or failed: paused at the entry
        self.paused = True
        return bytes([host.OK])

    def _c_N(self, args: bytes) -> bytes:
        if not self.paused:
            return bytes([host.BAD])
        self._one_step()
        if self.m.status != RUNNING:
            self.paused = False
        return bytes([host.OK])


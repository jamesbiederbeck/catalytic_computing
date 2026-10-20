"""Host side of the cmtz-core UART protocol.

Protocol, 8N1. Every command is one ASCII byte, then its arguments; multi-
byte numbers are big endian. Every field value fits a byte because q <= 254,
so 0xFF is never a legal value and marks errors.

  'I'  info, 21 bytes: "CMTC", ISA version, p, q, program memory size (2),
       data memory size (2), core hash (4), loaded program length (2),
       crc32 of the loaded program words (4)
  'P'  n(2) entry(2) handler(2, 0xFFFF = none), then n words of 4 bytes.
       Reply 5 bytes: 0x00 and the crc32 the core computed over the word
       bytes, or 0xFF x 5 (core running, n = 0 or too big, entry or
       handler outside the program; the words are consumed and dropped)
  'W'  r, v: write field register r. Reply 0x00, or 0xFF (running, r >= 32,
       v >= q)
  'D'  addr(2) count(2) then count bytes: write data memory. Reply 0x00, or
       0xFF (running, out of range, a byte >= q; nothing is written)
  'M'  addr(2) count(2): read data memory. Reply count bytes, or count x 0xFF
  'S'  registers, 64 bytes: R0..R31, then X0..X15 (2 bytes each), or all
       0xFF while running
  'G'  run from the entry. Reply 0x00 at once, or 0xFF (running, nothing
       loaded); poll with 'Q'
  'Q'  status, 16 bytes, any time: status, error, retired(4),
       interrupt count(2), retired at the last dispatch(4), pc saved by the
       last dispatch(2), pc(2)
  'T'  interrupt trigger. Reply 0x00
  'X'  abort a run (status error, code 7). Reply 0x00
  other bytes are ignored. A command that stops partway for TIMEOUT_CLKS is
  dropped; bytes sent while a reply goes out are ignored.

status: 0 idle, 1 running, 2 halted, 3 error. The run contract is the one in
`sim.py`: 'G' resets the control state only; registers and data memory keep
their contents.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

from .image import CoreConfig, ProgramImage

INFO_LEN, STATUS_LEN, REGS_LEN = 21, 16, 64
OK, BAD = 0x00, 0xFF


def words_crc(words: list[int]) -> int:
    return zlib.crc32(b"".join(w.to_bytes(4, "big") for w in words))


def be(v: int, n: int) -> bytes:
    return v.to_bytes(n, "big")


# ── encoders (shared with the testbench generator) ───────────────────────────

def cmd_info() -> bytes:
    return b"I"


def cmd_load(image: ProgramImage) -> bytes:
    h = 0xFFFF if image.handler is None else image.handler
    return (b"P" + be(len(image.words), 2) + be(image.entry, 2) + be(h, 2)
            + b"".join(be(w, 4) for w in image.words))


def cmd_write_reg(r: int, v: int) -> bytes:
    return b"W" + bytes([r, v])


def cmd_write_data(addr: int, values: list[int]) -> bytes:
    return b"D" + be(addr, 2) + be(len(values), 2) + bytes(values)


def cmd_read_data(addr: int, count: int) -> bytes:
    return b"M" + be(addr, 2) + be(count, 2)


def info_frame(core: CoreConfig, image: ProgramImage | None) -> bytes:
    n = len(image.words) if image else 0
    crc = words_crc(image.words) if image else 0
    return (b"CMTC" + bytes([core.isa_version, core.p, core.q])
            + be(core.pmem_size, 2) + be(core.dmem_size, 2)
            + be(core.core_hash, 4) + be(n, 2) + be(crc, 4))


@dataclass
class Status:
    status: int
    error: int
    retired: int
    interrupts: int
    disp_retired: int
    disp_pc: int
    pc: int

    @classmethod
    def parse(cls, b: bytes) -> Status:
        if len(b) != STATUS_LEN:
            raise ValueError(f"status frame is {len(b)} bytes")
        g = lambda i, n: int.from_bytes(b[i:i + n], "big")
        return cls(b[0], b[1], g(2, 4), g(6, 2), g(8, 4), g(12, 2), g(14, 2))

    def frame(self) -> bytes:
        return (bytes([self.status, self.error]) + be(self.retired, 4)
                + be(self.interrupts, 2) + be(self.disp_retired, 4)
                + be(self.disp_pc, 2) + be(self.pc, 2))


def parse_regs(b: bytes) -> tuple[list[int], list[int]]:
    return list(b[:32]), [int.from_bytes(b[32 + 2 * i:34 + 2 * i], "big")
                          for i in range(16)]


# ── a link to a board ────────────────────────────────────────────────────────

class CoreLink:
    """Drive a cmtz-core over a byte transport with `write(bytes)` and
    `read(n) -> bytes` (pyserial's Serial works)."""

    def __init__(self, port, core: CoreConfig, timeout: float = 2.0):
        self.port = port
        self.core = core
        self.timeout = timeout

    def _xfer(self, data: bytes, n: int) -> bytes:
        self.port.write(data)
        out = b""
        t0 = time.monotonic()
        while len(out) < n:
            chunk = self.port.read(n - len(out))
            if chunk:
                out += chunk
            elif time.monotonic() - t0 > self.timeout:
                raise TimeoutError(f"{len(out)} of {n} reply bytes after "
                                   f"{data[:1]!r}")
        return out

    def info(self) -> bytes:
        return self._xfer(cmd_info(), INFO_LEN)

    def check_core(self) -> None:
        got = self.info()
        want = info_frame(self.core, None)
        if got[:15] != want[:15]:
            raise RuntimeError(f"board is not this core: {got[:15].hex()} "
                               f"(expected {want[:15].hex()})")

    def load(self, image: ProgramImage) -> None:
        r = self._xfer(cmd_load(image), 5)
        if r[0] != OK or int.from_bytes(r[1:], "big") != words_crc(image.words):
            raise RuntimeError(f"program load failed: {r.hex()}")

    def write_reg(self, r: int, v: int) -> None:
        if self._xfer(cmd_write_reg(r, v), 1)[0] != OK:
            raise RuntimeError(f"W R{r} <- {v} refused")

    def write_data(self, addr: int, values: list[int]) -> None:
        for i in range(0, len(values), 256):
            chunk = values[i:i + 256]
            if self._xfer(cmd_write_data(addr + i, chunk), 1)[0] != OK:
                raise RuntimeError(f"D at {addr + i} refused")

    def read_data(self, addr: int, count: int) -> list[int]:
        out: list[int] = []
        for i in range(0, count, 256):
            n = min(256, count - i)
            out += list(self._xfer(cmd_read_data(addr + i, n), n))
        return out

    def regs(self) -> tuple[list[int], list[int]]:
        return parse_regs(self._xfer(b"S", REGS_LEN))

    def status(self) -> Status:
        return Status.parse(self._xfer(b"Q", STATUS_LEN))

    def go(self) -> None:
        if self._xfer(b"G", 1)[0] != OK:
            raise RuntimeError("G refused")

    def trigger(self) -> None:
        self._xfer(b"T", 1)

    def abort(self) -> None:
        self._xfer(b"X", 1)

    def wait(self, timeout: float = 10.0) -> Status:
        t0 = time.monotonic()
        while True:
            s = self.status()
            if s.status != 1 or time.monotonic() - t0 > timeout:
                return s

    def run(self, image: ProgramImage, inputs: dict[str, int] | None = None,
            matrices: dict[str, list[list[int]]] | None = None,
            trigger_after: float | None = None, load: bool = True) -> Status:
        """Load, write inputs and matrices, run to completion. With
        `trigger_after` (seconds), send one interrupt that long after `G`.

        Inputs and matrix entries must be field values in [0, p), as in
        `Machine.set_inputs`: the core's W and D accept anything below q,
        and the catalytic proofs say nothing about values in [p, q). Use
        `write_reg` directly to put such a value in a register on purpose."""
        bad = [f"{k}={v}" for k, v in (inputs or {}).items()
               if not 0 <= v < self.core.p]
        bad += [k for k, m in (matrices or {}).items()
                if any(not 0 <= v < self.core.p for row in m for v in row)]
        if bad:
            raise ValueError(f"values outside [0, p): {', '.join(bad)}")
        if load:
            self.load(image)
        for k, v in (inputs or {}).items():
            self.write_reg(image.inputs[k], v)
        for k, m in (matrices or {}).items():
            self.write_data(image.matrices_in[k].addr, sum(m, []))
        self.go()
        if trigger_after is not None:
            time.sleep(trigger_after)
            self.trigger()
        return self.wait()

    def outputs(self, image: ProgramImage) -> dict[str, int]:
        R, _ = self.regs()
        return {k: R[r] for k, r in image.outputs.items()}

    def read_matrix(self, image: ProgramImage, name: str) -> list[list[int]]:
        slot = image.matrices_out.get(name) or image.matrices_in[name]
        flat = self.read_data(slot.addr, slot.n * slot.n)
        return [flat[i * slot.n:(i + 1) * slot.n] for i in range(slot.n)]

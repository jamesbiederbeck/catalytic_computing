"""Host side of the multilane cmtz-core UART protocol (CMTM), RFC-003.

A separate protocol from the single core's (`host.py`, magic CMTC, which is
unchanged): a multilane bitstream answers 'I' with "CMTM", so the
single-core tools refuse it and vice versa. 8N1, big endian, 0xFF marks
errors (q <= 254). Commands act on the *selected lane* unless stated:

  'I'  info, 25 bytes: "CMTM", protocol version, lanes, leases (0/1), ISA
       version, p, q, program memory (2), data memory (2), multilane hash
       (4), selected lane, its loaded program length (2) and word crc32 (4)
  'L'  n: select lane n. Reply 0x00, or 0xFF (n >= lanes)
  'P' 'W' 'D' 'M' 'S' 'G' 'T' 'X'  as in host.py, on the selected lane
       (W is also refused for a register leased out or poisoned)
  'B'  mask: start every lane in mask on the same clock. Reply 0x00, or
       0xFF (empty mask, a lane that does not exist, is running or has no
       program)
  'Q'  status, 28 bytes: the 16 of host.py, then the lane's cycle stamps
       of its last run: first running cycle (4), first cycle after it (4),
       and the cycles it spent stalled on a lease (4). end - start is the
       run's cycle count.
  'K'  b, o, bank, count: lease the owner o's registers 8 bank .. 8 bank +
       count - 1 (bank 1..3, count 1..8) to borrower b, whose R24.. then
       name them. Both lanes idle. Reply 0x00; 0xFE if one of them holds
       a value outside [0, p) (nothing is leased); 0xFF otherwise (a lane
       would lend and borrow, the bank is taken, b already has a lease).
       count = 0 clears b's lease, active (unused) or poisoned, with b and
       its owner idle: the host's recovery after it restored the values.
  'E'  lease of the selected lane as borrower, 4 bytes: state (0 none,
       1 active, 2 poisoned), owner, bank, count

A borrower's lease ends when its run ends: released if it halted,
poisoned otherwise (the owner stalls on a poisoned register; abort it with
'X' and clear with K count 0 after restoring).
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

from .host import OK, BAD, CoreLink, be, cmd_load, cmd_read_data, \
    cmd_write_data, cmd_write_reg, parse_regs, words_crc
from .image import CoreConfig, ProgramImage

PROTO_VERSION = 1
INFO_FIXED, INFO_LEN, STATUS_LEN = 18, 25, 28
RANGE = 0xFE


def mlane_hash(core: CoreConfig, lanes: int, lease: bool) -> int:
    key = (f"mlane:{PROTO_VERSION}:{lanes}:{int(lease)}:{core.isa_version}:"
           f"{core.p}:{core.q}:{core.pmem_size}:{core.dmem_size}")
    return zlib.crc32(key.encode())


def info_frame(core: CoreConfig, lanes: int, lease: bool, sel: int,
               image: ProgramImage | None) -> bytes:
    n = len(image.words) if image else 0
    crc = words_crc(image.words) if image else 0
    return (b"CMTM" + bytes([PROTO_VERSION, lanes, int(lease),
                             core.isa_version, core.p, core.q])
            + be(core.pmem_size, 2) + be(core.dmem_size, 2)
            + be(mlane_hash(core, lanes, lease), 4)
            + bytes([sel]) + be(n, 2) + be(crc, 4))


def cmd_select(lane: int) -> bytes:
    return b"L" + bytes([lane])


def cmd_start(mask: int) -> bytes:
    return b"B" + bytes([mask])


def cmd_lease(b: int, o: int, bank: int, count: int) -> bytes:
    return b"K" + bytes([b, o, bank, count])


@dataclass
class LaneStatus:
    status: int
    error: int
    retired: int
    interrupts: int
    disp_retired: int
    disp_pc: int
    pc: int
    t_start: int
    t_end: int
    stalls: int

    @property
    def cycles(self) -> int:
        return (self.t_end - self.t_start) & 0xFFFFFFFF

    @classmethod
    def parse(cls, b: bytes) -> LaneStatus:
        if len(b) != STATUS_LEN:
            raise ValueError(f"status frame is {len(b)} bytes")
        g = lambda i, n: int.from_bytes(b[i:i + n], "big")
        return cls(b[0], b[1], g(2, 4), g(6, 2), g(8, 4), g(12, 2), g(14, 2),
                   g(16, 4), g(20, 4), g(24, 4))

    def frame(self) -> bytes:
        return (bytes([self.status, self.error]) + be(self.retired, 4)
                + be(self.interrupts, 2) + be(self.disp_retired, 4)
                + be(self.disp_pc, 2) + be(self.pc, 2) + be(self.t_start, 4)
                + be(self.t_end, 4) + be(self.stalls, 4))


@dataclass
class LeaseState:
    state: int
    owner: int
    bank: int
    count: int

    @classmethod
    def parse(cls, b: bytes) -> LeaseState:
        return cls(*b)


class MLaneLink(CoreLink):
    """Drive a multilane board. The single-core calls (load, write_reg,
    regs, go, run ...) act on the selected lane."""

    def __init__(self, port, core: CoreConfig, lanes: int, lease: bool,
                 timeout: float = 2.0):
        super().__init__(port, core, timeout)
        self.lanes, self.lease = lanes, lease
        self.sel = None

    def info(self) -> bytes:
        return self._xfer(b"I", INFO_LEN)

    def check_core(self) -> None:
        got = self.info()
        want = info_frame(self.core, self.lanes, self.lease, 0, None)
        if got[:INFO_FIXED] != want[:INFO_FIXED]:
            raise RuntimeError(f"board is not this multilane core: "
                               f"{got[:INFO_FIXED].hex()} (expected "
                               f"{want[:INFO_FIXED].hex()})")

    def select(self, lane: int) -> None:
        if self.sel == lane:
            return
        if self._xfer(cmd_select(lane), 1)[0] != OK:
            raise RuntimeError(f"L {lane} refused")
        self.sel = lane

    def start(self, mask: int) -> None:
        if self._xfer(cmd_start(mask), 1)[0] != OK:
            raise RuntimeError(f"B {mask:#x} refused")

    def lease_cmd(self, b: int, o: int, bank: int, count: int) -> int:
        return self._xfer(cmd_lease(b, o, bank, count), 1)[0]

    def lease_state(self) -> LeaseState:
        return LeaseState.parse(self._xfer(b"E", 4))

    def lane_status(self) -> LaneStatus:
        return LaneStatus.parse(self._xfer(b"Q", STATUS_LEN))

    def status(self):           # CoreLink.wait polls this
        return self.lane_status()

    def wait_all(self, lanes, timeout: float = 30.0) -> dict[int, LaneStatus]:
        t0 = time.monotonic()
        out = {}
        for lane in lanes:
            self.select(lane)
            while True:
                s = self.lane_status()
                if s.status != 1 or time.monotonic() - t0 > timeout:
                    out[lane] = s
                    break
        return out

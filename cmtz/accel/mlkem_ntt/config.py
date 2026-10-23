"""Accelerator configuration and pipeline timing constants."""
from dataclasses import dataclass

from .arith import Q, latency


@dataclass(frozen=True)
class NTTConfig:
    """One accelerator variant.  Everything the generator emits derives from this."""
    modulus: int = Q
    length: int = 256
    lanes: int = 1                  # parallel butterfly lanes (1, 2, 4, 8)
    reduction: str = "montgomery"   # or "barrett"
    memory: str = "banked"          # 2*lanes single-read/single-write banks, in place
    bank_ram: str = "block"         # "block": force BSRAM for the banks; "lut": let yosys use LUT RAM
    bank_prim: str = "infer"        # "infer" (yosys picks the BSRAM mode) or "sdp" (SDPX9B, READ_MODE=1)
    rom_ram: str = "block"          # "block" or "logic" (LUT ROM) for the control ROM
    lane: str = "deep"              # "deep" (closes timing at high clocks) or "compact" (6 stages)
    bin_reg: int = 1                # register the BSRAM address / write-enable / write-data inputs
    rd_reg: int = 2                 # read pipeline registers after the BSRAM read register (1st is absorbed into the BSRAM)
    rom_reg: int = 2                # registers after the ROM read (the first lets the BSRAM use its output register)
    xreg: int = 1                   # register stage on the read crossbar output
    wreg: int = 1                   # register stage on the write crossbar output
    clk_hz: int = 27_000_000        # clock the wrapper's UART divider and reports assume
    pll: tuple | None = None        # (idiv_sel, fbdiv_sel, odiv_sel) or None for the crystal

    def __post_init__(self):
        assert self.modulus == Q and self.length == 256, "ML-KEM only"
        assert self.lanes in (1, 2, 4, 8)
        assert self.reduction in ("montgomery", "barrett")
        assert self.memory == "banked"

    @property
    def banks(self):
        return 2 * self.lanes

    @property
    def bank_bits(self):
        return self.banks.bit_length() - 1

    @property
    def depth(self):
        return self.length // self.banks

    @property
    def addr_bits(self):
        return self.depth.bit_length() - 1

    # ---- pipeline timing, in clocks relative to the cycle t a read address is issued ----
    @property
    def lat(self):          # register stages in one butterfly lane
        return latency(self.reduction, self.lane)

    @property
    def rd_lat(self):       # clocks from a read address field (or host index) to data at the crossbar
        return 1 + self.bin_reg + self.rd_reg

    @property
    def t_ctl(self):        # crossbar select / twiddle / lane valid are applied this late
        return self.rd_lat
    @property
    def t_in(self):         # lane inputs (a, b, z) presented
        return self.t_ctl + self.xreg
    @property
    def t_out(self):        # lane outputs registered and visible
        return self.t_in + self.lat
    @property
    def t_wsel(self):       # write crossbar select applied
        return self.t_out
    @property
    def t_wr(self):         # bank write enable / address / data applied
        return self.t_out + self.wreg
    @property
    def dist(self):         # earliest issue cycle of a read that depends on this op's write
        return self.t_wr + self.bin_reg + 1

    @property
    def overhead(self):     # kernel cycles = control words + this (start/ROM latency and done)
        return 2 + self.rom_reg

    def tag(self):
        return f"l{self.lanes}-{self.reduction[:4]}"

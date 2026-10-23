"""Accelerator configuration and pipeline timing constants."""
from dataclasses import dataclass

from .arith import LAT, Q


@dataclass(frozen=True)
class NTTConfig:
    """One accelerator variant.  Everything the generator emits derives from this."""
    modulus: int = Q
    length: int = 256
    lanes: int = 1                  # parallel butterfly lanes (1, 2, 4, 8)
    reduction: str = "montgomery"   # or "barrett"
    memory: str = "banked"          # 2*lanes single-read/single-write banks, in place
    bank_ram: str = "block"         # "block": force BSRAM for the banks; "lut": let yosys use LUT RAM
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
    def t_ctl(self):        # crossbar select / twiddle / lane valid are applied this late
        return 1
    @property
    def t_in(self):         # lane inputs (a, b, z) presented
        return 1 + self.xreg
    @property
    def t_out(self):        # lane outputs registered and visible
        return self.t_in + LAT
    @property
    def t_wsel(self):       # write crossbar select applied
        return self.t_out
    @property
    def t_wr(self):         # bank write enable / address / data applied
        return self.t_out + self.wreg
    @property
    def dist(self):         # earliest issue cycle of a read that depends on this op's write
        return self.t_wr + 1

    def tag(self):
        return f"l{self.lanes}-{self.reduction[:4]}"

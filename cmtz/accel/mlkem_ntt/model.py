"""Cycle-level model of the accelerator datapath, driven only by the control words.

It mirrors the RTL register structure (sync-read banks, optional read/write crossbar
registers, ``LAT``-stage lanes) and uses ``arith.bfly_model`` for the lane arithmetic.
Running it on random data and comparing with the FIPS 203 oracle shows that the words
(addresses, selects, twiddles, spacing) are right; a read scheduled too early returns a
stale value and fails.
"""
from .arith import bfly_model
from .words import _blank
from .sched import BankMap


class Machine:
    def __init__(self, cfg, bank_map: BankMap):
        self.cfg, self.bm = cfg, bank_map
        self.mem = [[0] * cfg.depth for _ in range(cfg.banks)]

    def load(self, coeffs):
        for e, c in enumerate(coeffs):
            self.mem[self.bm.bank(e)][self.bm.addr(e)] = c

    def dump(self):
        return [self.mem[self.bm.bank(e)][self.bm.addr(e)] for e in range(256)]

    def run(self, words, inverse):
        cfg = self.cfg
        B, L = cfg.banks, cfg.lanes
        LAT = cfg.lat
        mode = 1 if inverse else 0
        rdata = [0] * B                  # bank read data seen by the crossbar
        rdp = [[0] * B for _ in range(cfg.rd_reg)]   # read pipeline registers after the BSRAM read register
        xin = [None] * L                 # lane input registers (if xreg)
        lane = [[None] * L for _ in range(LAT)]     # lane[s][l]: stage s+1 contents
        wreg = [0] * B
        ra_q = [0] * B                   # BSRAM input registers (when bin_reg)
        we_q, wa_q, wd_q = [0] * B, [0] * B, [0] * B
        # the BSRAM input register delays the last write one clock past the last word
        for w in list(words) + [_blank(cfg) for _ in range(cfg.bin_reg)]:
            # ---- combinational view of this cycle ----
            if cfg.xreg:
                lane_in = xin
            else:
                lane_in = [(rdata[w.asrc[l]], rdata[w.bsrc[l]], w.tw[l]) if w.lane_v[l] else None
                           for l in range(L)]
            lane_out = lane[LAT - 1]
            outs = [0] * (2 * L)
            for l in range(L):
                if lane_out[l] is not None:
                    outs[2 * l], outs[2 * l + 1] = lane_out[l]
            wdata = wreg if cfg.wreg else [outs[w.wsel[k]] for k in range(B)]
            # ---- state update at the clock edge ----
            ra, we, wa = (ra_q, we_q, wa_q) if cfg.bin_reg else (w.raddr, w.we, w.waddr)
            wd_now = wdata
            wd_in = wd_q if cfg.bin_reg else wdata
            new_rdm = [self.mem[k][ra[k]] for k in range(B)]            # read-first
            new_rdata = rdp[-1] if cfg.rd_reg else new_rdm
            for k in range(B):
                if we[k]:
                    self.mem[k][wa[k]] = wd_in[k]
            if cfg.bin_reg:
                ra_q, we_q, wa_q, wd_q = list(w.raddr), list(w.we), list(w.waddr), list(wd_now)
            if cfg.wreg:
                wreg = [outs[w.wsel[k]] for k in range(B)]
            for s in range(LAT - 1, 0, -1):
                lane[s] = lane[s - 1]
            lane[0] = [((lane_in[l][1], lane_in[l][0]) if cfg.lane == "swap" else
                        bfly_model(cfg.reduction, mode, *lane_in[l][:2], lane_in[l][2]))
                       if lane_in[l] is not None else None for l in range(L)]
            if cfg.xreg:
                xin = [(rdata[w.asrc[l]], rdata[w.bsrc[l]], w.tw[l]) if w.lane_v[l] else None
                       for l in range(L)]
            if cfg.rd_reg:
                rdp = [new_rdm] + rdp[:-1]
            rdata = new_rdata

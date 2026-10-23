"""Cycle-level model of the accelerator datapath, driven only by the control words.

It mirrors the RTL register structure (sync-read banks, optional read/write crossbar
registers, ``LAT``-stage lanes) and uses ``arith.bfly_model`` for the lane arithmetic.
Running it on random data and comparing with the FIPS 203 oracle shows that the words
(addresses, selects, twiddles, spacing) are right; a read scheduled too early returns a
stale value and fails.
"""
from .arith import LAT, bfly_model
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
        mode = 1 if inverse else 0
        rdata = [0] * B                  # BSRAM output registers
        xin = [None] * L                 # lane input registers (if xreg)
        lane = [[None] * L for _ in range(LAT)]     # lane[s][l]: stage s+1 contents
        wreg = [0] * B
        for w in words:
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
            new_rdata = [self.mem[k][w.raddr[k]] for k in range(B)]     # read-first
            for k in range(B):
                if w.we[k]:
                    self.mem[k][w.waddr[k]] = wdata[k]
            if cfg.wreg:
                wreg = [outs[w.wsel[k]] for k in range(B)]
            for s in range(LAT - 1, 0, -1):
                lane[s] = lane[s - 1]
            lane[0] = [bfly_model(cfg.reduction, mode, *lane_in[l][:2], lane_in[l][2])
                       if lane_in[l] is not None else None for l in range(L)]
            if cfg.xreg:
                xin = [(rdata[w.asrc[l]], rdata[w.bsrc[l]], w.tw[l]) if w.lane_v[l] else None
                       for l in range(L)]
            rdata = new_rdata

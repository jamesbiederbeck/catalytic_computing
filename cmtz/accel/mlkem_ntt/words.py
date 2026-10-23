"""Control words: the static schedule rendered cycle by cycle.

Every field is stored at the cycle the hardware consumes it, so the RTL needs no delay
lines: it just walks the table.  Field timing (relative to the issue cycle t of an op):

  raddr[bank]            t            read address of each bank
  lane_v/asrc/bsrc/tw    t_ctl        read-crossbar selects, twiddle, lane valid
  wsel[bank]             t_wsel       write-crossbar select (which lane output)
  we/waddr[bank]         t_wr         bank write enable and address
"""
from dataclasses import dataclass, field

from .config import NTTConfig
from .sched import Schedule
from .twiddles import table


@dataclass
class Word:
    raddr: list
    lane_v: list
    asrc: list
    bsrc: list
    tw: list
    wsel: list
    we: list
    waddr: list


def _blank(cfg):
    B, L = cfg.banks, cfg.lanes
    return Word([0] * B, [0] * L, [0] * L, [0] * L, [0] * L, [0] * B, [0] * B, [0] * B)


def build_words(s: Schedule):
    cfg, bm = s.cfg, s.bank_map
    tw = table(cfg.reduction, s.inverse)
    words = [_blank(cfg) for _ in range(s.words)]
    for t, idxs in s.issue.items():
        for lane, i in enumerate(idxs):
            op = s.ops[i]
            ba, bb = bm.bank(op.a), bm.bank(op.b)
            aa, ab = bm.addr(op.a), bm.addr(op.b)
            words[t].raddr[ba], words[t].raddr[bb] = aa, ab
            w = words[t + cfg.t_ctl]
            w.lane_v[lane], w.asrc[lane], w.bsrc[lane], w.tw[lane] = 1, ba, bb, tw[op.k]
            ws = words[t + cfg.t_wsel]
            ws.wsel[ba], ws.wsel[bb] = 2 * lane, 2 * lane + 1
            wr = words[t + cfg.t_wr]
            wr.we[ba] = wr.we[bb] = 1
            wr.waddr[ba], wr.waddr[bb] = aa, ab
    return words

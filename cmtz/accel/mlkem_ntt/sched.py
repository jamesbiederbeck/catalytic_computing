"""Static scheduler for the banked, in-place NTT accelerator.

Arithmetic semantics (what a butterfly is) live in ``arith``; this module only decides
*when* and *where* butterflies run.  It models the 7 x 128 butterflies of FIPS 203
Algorithms 9/10, assigns them to lanes and clock cycles, and maps coefficients onto
2*lanes single-read/single-write RAM banks.

Hazards.  The transform is in place, so every coefficient is touched by a chain of seven
butterflies, each reading and writing the same element.  The only hazard is therefore
read-after-write along that chain: a butterfly may issue ``cfg.dist`` cycles after the
one that produced its operands.  Bank conflicts are the other constraint: the 2*lanes
operands of one cycle must live in distinct banks (one read port per bank).  Writes land
a fixed delay after their reads, from the same banks, so they cannot conflict either; the
checker in ``verify`` proves that rather than assuming it.
"""
from dataclasses import dataclass

from .config import NTTConfig

N = 256


def _bitrev7_by_table(k):
    # independent of the oracle: build the reversed index by shifting bits
    r = 0
    for _ in range(7):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


@dataclass(frozen=True)
class Op:
    stage: int
    a: int          # element read as the "top" operand (index j)
    b: int          # element read as the "bottom" operand (index j + len)
    k: int          # zeta index (value is zeta^BitRev7(k))


def spec_ops(inverse):
    """The 896 butterflies of Algorithm 9 (forward) or 10 (inverse), in spec order."""
    ops = []
    if not inverse:
        k, length, stage = 1, 128, 0
        while length >= 2:
            for start in range(0, N, 2 * length):
                for j in range(start, start + length):
                    ops.append(Op(stage, j, j + length, k))
                k += 1
            length //= 2
            stage += 1
    else:
        k, length, stage = 127, 2, 0
        while length <= 128:
            for start in range(0, N, 2 * length):
                for j in range(start, start + length):
                    ops.append(Op(stage, j, j + length, k))
                k -= 1
            length *= 2
            stage += 1
    return ops


# ---------------------------------------------------------------------------- banking

class BankMap:
    """Linear (GF(2)) map from a coefficient index to (bank, address).

    ``cols[i]`` is the contribution of index bit i to the bank number; the bank is the
    XOR of the columns of the set bits.  ``pivots`` are bit positions whose columns form
    a basis, so the remaining bits (the address) plus the bank determine the index.
    """

    def __init__(self, bank_bits, cols):
        self.bank_bits = bank_bits
        self.cols = tuple(cols)
        self.pivots = self._basis()
        self.addr_pos = [i for i in range(8) if i not in self.pivots]

    def _basis(self):
        chosen, span = [], {0}
        for i in range(8):
            c = self.cols[i]
            if c not in span:
                chosen.append(i)
                span |= {s ^ c for s in span}
            if len(chosen) == self.bank_bits:
                return tuple(chosen)
        raise ValueError("columns do not span the bank space")

    def bank(self, e):
        b = 0
        for i in range(8):
            if (e >> i) & 1:
                b ^= self.cols[i]
        return b

    def addr(self, e):
        a = 0
        for n, i in enumerate(self.addr_pos):
            a |= ((e >> i) & 1) << n
        return a


def default_bank_map(lanes):
    """Bank maps found by ``search_bank_map`` (kept here so generation is search-free).

    Tuples are the per-bit columns.  Regenerate with ``search_bank_map`` if the pipeline
    or priority rule changes; ``verify.check_bank_legality`` re-proves whatever is used.
    """
    return BankMap(_BITS[lanes], _COLS[lanes])


_BITS = {1: 1, 2: 2, 4: 3, 8: 4}
_COLS = {
    1: (1, 1, 1, 1, 1, 1, 1, 1),
    2: (1, 2, 3, 1, 2, 3, 1, 2),
    4: (1, 2, 4, 3, 5, 6, 7, 1),
    8: (1, 2, 4, 8, 3, 5, 9, 7),
}


# ------------------------------------------------------------------------- scheduling

@dataclass
class Schedule:
    cfg: NTTConfig
    inverse: bool
    bank_map: BankMap
    ops: list              # spec-ordered Op list
    issue: dict            # cycle -> [op index, ...] (lane l = position in the list)
    issue_cycle: list      # op index -> cycle
    words: int             # number of control words (cycles up to the last write)

    @property
    def last_issue(self):
        return max(self.issue)


def schedule(cfg, inverse, bank_map=None, priority=None):
    bm = bank_map or default_bank_map(cfg.lanes)
    ops = spec_ops(inverse)
    order = list(range(len(ops)))
    if priority is not None:        # exploration / mutation hook; default is spec order
        order.sort(key=lambda i: priority(ops[i], i))
    dist = cfg.dist
    L = cfg.lanes
    # op chain per element: indices of ops touching it, in spec order
    chain = [[] for _ in range(N)]
    for i, op in enumerate(ops):
        chain[op.a].append(i)
        chain[op.b].append(i)
    nxt = [0] * N              # position in each element's chain
    ready_at = [0] * N         # cycle at which the element's latest value is readable
    done = [False] * len(ops)
    issue, issue_cycle = {}, [None] * len(ops)
    remaining = len(ops)
    # candidate ops are those at the head of both operands' chains; kept in spec order
    t = 0
    window = []
    cursor = 0                 # next op index not yet placed in the window
    while remaining:
        # grow window: spec-order ops that could become eligible
        while cursor < len(ops):
            window.append(order[cursor])
            cursor += 1
        picked, used = [], set()
        for i in window:
            if len(picked) == L:
                break
            op = ops[i]
            if chain[op.a][nxt[op.a]] != i or chain[op.b][nxt[op.b]] != i:
                continue
            if ready_at[op.a] > t or ready_at[op.b] > t:
                continue
            ba, bb = bm.bank(op.a), bm.bank(op.b)
            if ba in used or bb in used or ba == bb:
                continue
            used.update((ba, bb))
            picked.append(i)
        for i in picked:
            op = ops[i]
            done[i] = True
            issue_cycle[i] = t
            nxt[op.a] += 1
            nxt[op.b] += 1
            ready_at[op.a] = ready_at[op.b] = t + dist
            remaining -= 1
        if picked:
            issue[t] = picked
            window = [i for i in window if not done[i]]
        t += 1
        if t > 100000:
            raise RuntimeError("scheduler made no progress")
    words = max(issue) + cfg.t_wr + 1
    return Schedule(cfg, inverse, bm, ops, issue, issue_cycle, words)


def search_bank_map(cfg, iters=400, seed=1):
    """Deterministic hill-climb over linear bank maps, minimising forward+inverse words."""
    import random
    rng = random.Random(seed)
    bits = cfg.bank_bits

    def cost(cols):
        try:
            bm = BankMap(bits, cols)
            return schedule(cfg, False, bm).words + schedule(cfg, True, bm).words
        except ValueError:
            return 10 ** 9

    best = list(default_bank_map(cfg.lanes).cols)
    best_c = cost(best)
    for _ in range(iters):
        cand = list(best)
        for _ in range(rng.choice((1, 1, 2))):
            cand[rng.randrange(8)] = rng.randrange(1, 1 << bits)
        c = cost(cand)
        if c <= best_c:
            best, best_c = cand, c
    return tuple(best), best_c

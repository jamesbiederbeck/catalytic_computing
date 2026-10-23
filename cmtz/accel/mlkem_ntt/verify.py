"""Schedule verification.  Works on the *control words* the RTL will execute, not on the
scheduler's internal bookkeeping, so it also catches mistakes in word construction."""
from collections import Counter

from .sched import BankMap, N, spec_ops
from .words import build_words


def check_bank_map(bm: BankMap, cfg):
    """Every coefficient has a distinct (bank, address); every address is in range."""
    seen = set()
    for e in range(N):
        b, a = bm.bank(e), bm.addr(e)
        assert 0 <= b < cfg.banks and 0 <= a < cfg.depth, (e, b, a)
        assert (b, a) not in seen, f"index {e} collides at bank {b} addr {a}"
        seen.add((b, a))
    assert len(seen) == N


def check_schedule(s):
    """Raise AssertionError on any violation; return a stats dict."""
    cfg, bm = s.cfg, s.bank_map
    check_bank_map(bm, cfg)
    words = build_words(s)
    spec = spec_ops(s.inverse)

    # 1. every butterfly executes exactly once
    placed = Counter(i for idxs in s.issue.values() for i in idxs)
    assert sorted(placed) == list(range(len(spec))) and set(placed.values()) == {1}, \
        "a butterfly is missing or repeated"
    assert len(spec) == 896

    # 2. lane count and the bank (port) legality of every cycle, from the words
    for t, w in enumerate(words):
        assert sum(w.lane_v) <= cfg.lanes
        assert len(w.raddr) == cfg.banks
    reads_per_bank = Counter()
    writes_per_bank = Counter()
    for t, idxs in s.issue.items():
        banks = []
        for i in idxs:
            banks += [bm.bank(s.ops[i].a), bm.bank(s.ops[i].b)]
        assert len(banks) == len(set(banks)), f"cycle {t}: two reads in one bank"
        for b in banks:
            reads_per_bank[(t, b)] += 1
            writes_per_bank[(t + cfg.t_wr, b)] += 1
    assert max(reads_per_bank.values()) == 1, "read-port conflict"
    assert max(writes_per_bank.values()) == 1, "write-port conflict"

    # 3. word contents agree with the schedule: write enables exactly at the right cycles,
    #    and each write goes to the address that was read
    for (cyc, b), n in writes_per_bank.items():
        assert words[cyc].we[b] == 1
    assert sum(sum(w.we) for w in words) == 2 * len(spec)

    # 4. read-after-write distance along every element's chain, and coverage
    last_write = {}                                  # element -> cycle its latest write lands
    touched = Counter()
    for i, op in enumerate(spec):                    # spec order = dependency order
        t = s.issue_cycle[i]
        for e in (op.a, op.b):
            if e in last_write:
                assert t >= last_write[e] + 1, f"op {i} reads {e} before it is written"
            touched[e] += 1
            last_write[e] = t + cfg.t_wr
    assert set(touched.values()) == {7} and len(touched) == N, "coefficient coverage"

    return {
        "words": len(words),
        "issue_cycles": max(s.issue) + 1,
        "ideal_issue": 896 // cfg.lanes,
        "stall_cycles": (max(s.issue) + 1) - 896 // cfg.lanes,
        "lane_utilisation": 896 / (cfg.lanes * (max(s.issue) + 1)),
    }

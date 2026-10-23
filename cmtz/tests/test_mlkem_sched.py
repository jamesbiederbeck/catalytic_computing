"""Scheduler, control words, and cycle model for the ML-KEM NTT accelerator."""
import random

import pytest

from cmtz.accel.mlkem_ntt import fips203_oracle as O
from cmtz.accel.mlkem_ntt import model, sched, twiddles, verify, words
from cmtz.accel.mlkem_ntt.config import NTTConfig

LANES = [1, 2, 4, 8]


ELEMENT_MAJOR = lambda op, i: (op.a, op.stage)     # adversarial: dependents issue back to back


class NoHazard(NTTConfig):
    """Scheduler that ignores the write latency: used to prove the checks bite."""
    @property
    def dist(self):
        return 1


def vectors():
    rng = random.Random(7)
    v = {"zeros": [0] * 256, "max": [3328] * 256, "ones": [1] * 256,
         "ramp": [i * 13 % 3329 for i in range(256)],
         "alt": [3328 if i % 2 else 0 for i in range(256)],
         "impulse0": [1] + [0] * 255, "impulse255": [0] * 255 + [1],
         "impulse_mid": [0] * 100 + [3328] + [0] * 155}
    for n in range(4):
        v[f"rand{n}"] = [rng.randrange(3329) for _ in range(256)]
    return v


def run(cfg, inverse, f):
    s = sched.schedule(cfg, inverse)
    m = model.Machine(cfg, s.bank_map)
    m.load(f)
    m.run(words.build_words(s), inverse)
    return m.dump()


class TestTwiddles:
    def test_independent_tables_agree_with_oracle(self):
        assert twiddles.ZETA == O.ZETAS


class TestSchedule:
    @pytest.mark.parametrize("lanes", LANES)
    @pytest.mark.parametrize("inverse", [False, True])
    def test_verifier_passes(self, lanes, inverse):
        cfg = NTTConfig(lanes=lanes)
        st = verify.check_schedule(sched.schedule(cfg, inverse))
        assert st["stall_cycles"] <= (2 if lanes == 8 else 0)

    @pytest.mark.parametrize("lanes", LANES)
    def test_spec_has_896_butterflies(self, lanes):
        assert len(sched.spec_ops(False)) == len(sched.spec_ops(True)) == 896

    def test_bank_map_must_be_bijective(self):
        with pytest.raises(ValueError):
            sched.BankMap(2, (1,) * 8)                       # columns do not span 4 banks
        cfg = NTTConfig(lanes=2)
        bm = sched.BankMap(2, (1, 2, 3, 1, 2, 3, 1, 2))
        verify.check_bank_map(bm, cfg)                       # a good map passes
        bad = sched.BankMap(2, (1, 2, 3, 1, 2, 3, 1, 2))
        bad.addr_pos = bad.addr_pos[:-1] + [bad.addr_pos[0]]  # address loses a bit
        with pytest.raises(AssertionError):
            verify.check_bank_map(bad, cfg)

    def test_verifier_catches_bank_conflict(self):
        s = sched.schedule(NTTConfig(lanes=2), False)
        t = sorted(s.issue)[3]
        # force an op that shares a bank with one already in that cycle
        first = s.ops[s.issue[t][0]]
        clash = next(i for i, op in enumerate(s.ops) if i not in s.issue[t]
                     and s.bank_map.bank(op.a) == s.bank_map.bank(first.a))
        old = s.issue_cycle[clash]
        s.issue[old].remove(clash)
        if not s.issue[old]:
            del s.issue[old]
        s.issue[t] = s.issue[t][:1] + [clash]
        s.issue_cycle[clash] = t
        with pytest.raises(AssertionError):
            verify.check_schedule(s)

    def test_verifier_catches_missing_butterfly(self):
        s = sched.schedule(NTTConfig(lanes=4), False)
        t = sorted(s.issue)[10]
        del s.issue[t][0]
        with pytest.raises(AssertionError):
            verify.check_schedule(s)

    def test_verifier_catches_raw_violation(self):
        s = sched.schedule(NoHazard(lanes=4), False, priority=ELEMENT_MAJOR)
        s.cfg = NTTConfig(lanes=4)                           # hardware keeps true timing
        with pytest.raises(AssertionError):
            verify.check_schedule(s)

    @pytest.mark.parametrize("lanes", LANES)
    def test_report_minimum_slack(self, lanes):
        """Smallest gap between a producer's write and its consumer's read; >= 1 is legal."""
        cfg = NTTConfig(lanes=lanes)
        s = sched.schedule(cfg, False)
        last = {}
        slack = 10 ** 9
        for i, op in enumerate(s.ops):
            t = s.issue_cycle[i]
            for e in (op.a, op.b):
                if e in last:
                    slack = min(slack, t - (last[e] + 1))
                last[e] = t + cfg.t_wr
        assert slack >= 0


class TestCycleModel:
    @pytest.mark.parametrize("lanes", LANES)
    @pytest.mark.parametrize("reduction", ["montgomery", "barrett"])
    def test_matches_oracle(self, lanes, reduction):
        cfg = NTTConfig(lanes=lanes, reduction=reduction)
        for name, f in vectors().items():
            assert run(cfg, False, f) == O.ntt(f), (name, "forward")
            assert run(cfg, True, f) == O.ntt_inv(f), (name, "inverse")

    @pytest.mark.parametrize("lanes", [1, 4])
    def test_forward_then_inverse_is_identity(self, lanes):
        cfg = NTTConfig(lanes=lanes)
        for name, f in vectors().items():
            assert run(cfg, True, run(cfg, False, f)) == f, name

    def test_too_early_read_gives_wrong_answer(self):
        cfg = NTTConfig(lanes=4)
        f = vectors()["rand0"]
        s = sched.schedule(NoHazard(lanes=4), False, priority=ELEMENT_MAJOR)
        s.cfg = cfg
        m = model.Machine(cfg, s.bank_map)
        m.load(f)
        m.run(words.build_words(s), False)
        assert m.dump() != O.ntt(f)

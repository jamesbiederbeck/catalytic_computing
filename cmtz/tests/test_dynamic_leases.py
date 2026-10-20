"""Dynamic leases (RFC-003 A4, simulator only): BORROW / RELEASE."""

import random

import pytest

from cmtz.analysis.isa_clean import CleanProofError, run_symbolic
from cmtz.analysis.lease_invariants import check_trace
from cmtz.analysis.lease_proof import Checker, LeaseRefused, certify_dynamic
from cmtz.isa import projection as PR
from cmtz.isa import workloads as W
from cmtz.isa.asm import AsmError, assemble
from cmtz.isa.encoding import OPS, decode
from cmtz.isa.image import ProgramImage
from cmtz.isa.multicore import (
    E_ABORT, E_LEASE, Job, LaneConfig, MultiCore, Variant,
)
from cmtz.isa.sim import E_OPCODE, ERROR, HALTED, Machine

CORE = PR.CORE
P = CORE.p
G = 307 * 2                     # gate_clean cycles


def img(src, inputs=None, outputs=None):
    a = assemble(src, ext=True)
    return ProgramImage(P, CORE.q, a.words, a.entry, a.handler,
                        inputs=inputs or {}, outputs=outputs or {},
                        listing=[str(i) for i in a.instrs])


def live(mc, lane, regs=(16, 17, 18), vals=(3, 9, 12)):
    for r, v in zip(regs, vals):
        mc.write_reg(lane, r, v)


class TestEncoding:
    def test_not_part_of_the_single_core_isa(self):
        assert "BORROW" not in OPS and "RELEASE" not in OPS
        with pytest.raises(AsmError):
            assemble("BORROW 0, 2, 3")
        w = assemble("BORROW 1, 2, 3", ext=True).words[0]
        with pytest.raises(ValueError):
            decode(w)
        assert str(decode(w, ext=True)) == "BORROW 1, 2, 3"

    def test_single_core_reports_a_bad_opcode(self):
        image = W.gate_dynamic(0, 2)
        with pytest.raises(ValueError):
            Machine(CORE, image)                # the single core's loader
        m = Machine(CORE, image, ext=True)
        m.run()
        assert (m.status, m.error, m.retired) == (ERROR, E_OPCODE, 0)


class TestProof:
    def test_certifies(self):
        c = certify_dynamic(W.gate_dynamic(0, 2, 3), W.gate_dynamic_claim())
        assert c.borrows == ((0, 2, 3),) and c.retired == 307 + 2 + 3

    @pytest.mark.parametrize("src, why", [
        ("BORROW 0, 2, 1\nADD R24, R24, R1\nRELEASE\nHALT", "when the lease ends"),
        ("BORROW 0, 2, 1\nADD R24, R24, R1\nHALT", "when the lease ends"),
        ("BORROW 0, 2, 1\nBORROW 0, 2, 1\nHALT", "while holding"),
        ("RELEASE\nHALT", "without a lease"),
        ("BORROW 0, 2, 1\nBZ R24, x\nx: RELEASE\nHALT", "depends on the inputs"),
        ("BORROW 0, 2, 1\nST R24, [X0+0]\nRELEASE\nHALT", "not supported"),
    ])
    def test_refuses(self, src, why):
        with pytest.raises(LeaseRefused) as e:
            certify_dynamic(img(src, {"v": 1}), W.gate_dynamic_claim().__class__.of({}))
        errs = e.value.violations
        if errs == ["no BORROW in the program"]:
            with pytest.raises(CleanProofError) as e2:
                run_symbolic(img(src), ext=True)
            errs = e2.value.violations
        assert any(why in v for v in errs), errs

    def test_private_window_values_come_back_after_release(self):
        # R24 is the lane's own before BORROW and after RELEASE
        run = run_symbolic(img("LDI R24, 5\nBORROW 0, 2, 1\nMULI R24, R24, 3\n"
                               "MULI R24, R24, 6\nRELEASE\nADD R1, R24, R0\nHALT"),
                           ext=True)
        assert run.R[1].constant() == 5 and run.borrows == [(0, 2, 1)]


class TestModel:
    def test_lease_lasts_from_borrow_to_release(self):
        """The owner, stalled on a leased register, resumes one cycle after
        the borrower's RELEASE, not at its HALT 40 NOPs later."""
        mc = MultiCore(CORE, 2, snapshot=True, trace_access=True)
        live(mc, 0)
        own = img("ADD R5, R16, R17\nHALT", outputs={"o": 5})
        mc.submit(W.dynamic_job("d", 0, 2, 6, 1, 0, 4, lane=1, tail=40),
                  Job("o", [Variant(own)], lane=0, arrival=10))
        rep = mc.run()
        r = rep.by_name()
        assert r["d"].outputs["y"] == 5 and r["d"].restored
        rel = next(e["t"] for e in mc.trace if e["ev"] == "release_req")
        assert r["o"].end == rel + 1 + 3      # resume, EXEC, FETCH, EXEC HALT
        assert r["o"].outputs == {"o": 12}
        assert r["d"].end > r["o"].end
        assert not check_trace(mc.trace)

    def test_simultaneous_requests_lower_lane_first(self):
        mc = MultiCore(CORE, 3, snapshot=True)
        live(mc, 0)
        ja = W.dynamic_job("a", 0, 2, 6, 1, 1, 0, lane=1)
        jb = W.dynamic_job("b", 0, 2, 8, 1, 1, 0, lane=2)
        mc.submit(ja, jb)
        rep = mc.run()
        a, b = rep.by_name()["a"], rep.by_name()["b"]
        assert a.stalls == 0 and b.stalls > 0
        # b's BORROW (EXEC at 1) retries at a's release, one cycle after
        # a's RELEASE (EXEC at a.end - 3: RELEASE, FETCH, EXEC HALT)
        assert b.stalls == a.end - 3 + 1 - 1
        assert a.outputs == ja.expect and b.outputs == jb.expect
        assert mc.regs(0)[16:19] == [3, 9, 12]
        assert not check_trace(mc.trace)

    def test_waiters_are_served_oldest_first(self):
        """Lane 1 holds the bank; lane 3 asks first, lane 2 later. At the
        release both retry in the same cycle; lane 3 (older) wins although
        lane 2 has the lower index."""
        mc = MultiCore(CORE, 4, snapshot=True)
        live(mc, 0)
        mc.submit(W.dynamic_job("h", 0, 2, 6, 1, 1, 0, lane=1),
                  W.dynamic_job("old", 0, 2, 6, 1, 1, 0, lane=3, arrival=5),
                  W.dynamic_job("new", 0, 2, 6, 1, 1, 0, lane=2, arrival=50))
        rep = mc.run()
        r = rep.by_name()
        order = [e["lane"] for e in mc.trace if e["ev"] == "acquire"]
        assert order == [1, 3, 2]
        assert r["new"].end > r["old"].end
        assert not check_trace(mc.trace)

    @pytest.mark.parametrize("src, why", [
        ("BORROW 1, 2, 3\nHALT", "BORROW from lane 1"),      # itself
        ("BORROW 5, 2, 3\nHALT", "BORROW from lane 5"),      # no such lane
        ("BORROW 0, 0, 3\nHALT", "bank 0"),
        ("BORROW 0, 2, 9\nHALT", "count 9"),
        ("BORROW 0, 2, 1\nBORROW 0, 3, 1\nHALT", "while holding"),
        ("RELEASE\nHALT", "without a lease"),
        ("BORROW 0, 2, 1\nRELEASE\nRELEASE\nHALT", "without a lease"),
    ])
    def test_invalid_requests_fault(self, src, why):
        mc = MultiCore(CORE, 2)
        mc.submit(Job("x", [Variant(img(src))], lane=1))
        r = mc.run().results[0]
        assert (r.status, r.error) == (ERROR, E_LEASE)
        assert any(why in e.get("why", "") for e in mc.trace if e["ev"] == "fault")
        assert not check_trace(mc.trace)

    def test_value_outside_the_field_faults(self):
        mc = MultiCore(CORE, 2)
        live(mc, 0, vals=(3, 18, 12))
        mc.submit(W.dynamic_job("d", 0, 2, 6, 1, 0, 4, lane=1))
        r = mc.run().results[0]
        assert (r.status, r.error) == (ERROR, E_LEASE)

    def test_borrower_fault_poisons(self):
        mc = MultiCore(CORE, 2, snapshot=True)
        live(mc, 0)
        bad = img("BORROW 0, 2, 3\nADD R24, R24, R1\nRET\nHALT", {"v": 1})
        rd = img("ADD R5, R16, R16\nHALT", outputs={"o": 5})
        mc.submit(Job("bad", [Variant(bad)], {"v": 2}, lane=1),
                  Job("rd", [Variant(rd)], lane=0, arrival=1))
        rep = mc.run()
        assert rep.leases["poison"] == 1
        assert rep.by_name()["rd"].error == E_ABORT
        assert not check_trace(mc.trace)

    def test_lender_that_borrows_waits_instead_of_deadlocking(self):
        """Lane 0 lends its bank 2 to lane 1 (static) and then asks for lane
        2's bank: it waits until lane 1 returns the lease, then proceeds."""
        mc = MultiCore(CORE, 3, policy="static", snapshot=True)
        live(mc, 0)
        live(mc, 2, vals=(1, 2, 3))
        mc.submit(W.static_gate("s", 0, (16, 17, 18), 6, 1, 0, 1, lane=1),
                  W.dynamic_job("d", 2, 2, 6, 0, 1, 0, lane=0, arrival=3))
        rep = mc.run()
        s, d = rep.by_name()["s"], rep.by_name()["d"]
        assert d.stalls == s.end - 4              # waited for lane 1's release
        assert s.outputs["y"] == 2 and d.outputs["y"] == 1
        assert mc.regs(0)[16:19] == [3, 9, 12] and mc.regs(2)[16:19] == [1, 2, 3]
        assert not check_trace(mc.trace)

    def test_owner_halts_while_lending(self):
        mc = MultiCore(CORE, 2, snapshot=True)
        live(mc, 0)
        quick = img("LDI R1, 1\nHALT")
        reader = img("ADD R5, R16, R16\nHALT", outputs={"o": 5})
        mc.submit(W.dynamic_job("d", 0, 2, 6, 1, 0, 4, lane=1),
                  Job("quick", [Variant(quick)], lane=0, arrival=3),
                  Job("rd", [Variant(reader)], lane=0, after=("quick",)))
        rep = mc.run()
        r = rep.by_name()
        assert r["quick"].stalls == 0 and r["quick"].end < r["d"].end
        assert r["rd"].stalls > 0 and r["rd"].outputs == {"o": 6}
        assert not check_trace(mc.trace)


class TestChecker:
    def test_unproven_program_is_not_dispatched(self):
        chk = Checker()
        mc = MultiCore(CORE, 2, accept=chk, snapshot=True)
        live(mc, 0)
        good = W.dynamic_job("good", 0, 2, 6, 1, 0, 4, lane=1)
        bad = W.dynamic_job("bad", 0, 2, 6, 1, 0, 4, lane=1, restore=False)
        mc.submit(bad, good)
        rep = mc.run()
        assert "bad" in mc.rejected and "bad" not in rep.by_name()
        assert rep.by_name()["good"].restored
        acq = [e for e in mc.trace if e["ev"] == "acquire"]
        assert len(acq) == 1 and acq[0]["cert"]

    @pytest.mark.parametrize("seed", range(4))
    def test_random_dynamic_mix(self, seed):
        rng = random.Random(seed)
        lanes = 4
        mc = MultiCore(CORE, lanes, accept=Checker(), snapshot=True,
                       seed=seed, randomize=True, jitter=20)
        for o in range(lanes):
            live(mc, o, vals=tuple(rng.randrange(P) for _ in range(3)))
        jobs = []
        for i in range(16):
            lane = rng.randrange(lanes)
            owner = rng.choice([o for o in range(lanes) if o != lane])
            jobs.append(W.dynamic_job(f"d{i}", owner, 2, rng.randrange(16),
                                      rng.randrange(2), rng.randrange(2),
                                      rng.randrange(P), lane=lane,
                                      tail=rng.randrange(20),
                                      arrival=rng.randrange(2000)))
        mc.submit(*jobs)
        before = {o: mc.regs(o)[16:19] for o in range(lanes)}
        rep = mc.run()
        assert not rep.pending and not check_trace(mc.trace)
        for j in jobs:
            r = rep.by_name()[j.name]
            assert r.status == HALTED and r.outputs["y"] == j.expect["y"]
        assert {o: mc.regs(o)[16:19] for o in range(lanes)} == before
        assert rep.leases["acquire"] == 16 == rep.leases["restored"]


class TestLeastCost:
    """The A4 policy: borrow, wait, spill or run privately, by estimated
    finish time."""

    def _mc(self, budget, lanes=2):
        from cmtz.analysis.lease_proof import Checker
        cfgs = [LaneConfig(lendable=(16, 17, 18, 19, 20))] + \
               [LaneConfig(private=budget)] * (lanes - 1)
        mc = MultiCore(CORE, lanes, cfgs, policy="cost", snapshot=True,
                       accept=Checker())
        mc.scramble(3, below=P)
        return mc

    def kinds(self, mc, jobs):
        mc.submit(*jobs)
        rep = mc.run()
        assert not rep.pending and not check_trace(mc.trace)
        for j in jobs:
            assert rep.by_name()[j.name].outputs["y"] == j.expect["y"]
        return [rep.by_name()[j.name].kind for j in jobs], rep

    def test_spill_beats_borrowing_when_it_fits(self):
        mc = self._mc(7)
        ks, _ = self.kinds(mc, [W.pressure_job(
            "g", "", 6, (1, 0), 3, ("leased", "spill", "direct"), lane=1)])
        assert ks == ["spill"]            # direct needs 8 registers

    def test_borrowing_when_nothing_else_fits(self):
        mc = self._mc(7)
        ks, rep = self.kinds(mc, [W.pressure_job(
            "g", "", 6, (1, 0), 3, ("leased", "clean", "direct"), lane=1)])
        assert ks == ["leased"] and rep.leases["restored"] == 1

    def test_private_when_it_fits(self):
        mc = self._mc(10)
        ks, _ = self.kinds(mc, [W.pressure_job(
            "g", "", 6, (1, 0), 3, ("leased", "clean"), lane=1)])
        assert ks == ["clean"]            # same cycles, no lease needed

    def test_waits_for_a_lease_rather_than_a_slow_private_run(self):
        """Lanes 1 and 2 both want a tree. Lane 0 lends one bank only, so
        the second tree either waits for the lease (39 910 cycles) or runs
        the private clean tree; with budget 17 only leased or spill fit."""
        mc = self._mc(17, lanes=3)
        jobs = [W.pressure_job(f"t{i}", "", (6, 8, 14), (1, 0, 1, 1), 2,
                               ("leased",), lane=1 + i) for i in range(2)]
        ks, rep = self.kinds(mc, jobs)
        assert ks == ["leased", "leased"]
        stalls = sorted(rep.by_name()[j.name].start for j in jobs)
        assert stalls[1] >= 39910         # the second waited for the first


def test_storage_benchmark_gate():
    """The A5 storage result for the gate stream, pinned: borrowing gets
    lane 1 down to the spill form's 7 registers, not below, and is 32x
    slower per gate (examples/fpga/mlane_bench.py)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "examples/fpga/mlane_bench.py"
    spec = importlib.util.spec_from_file_location("mlane_bench", path)
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    got = {}
    for name, (policy, forms) in B.POLICY_FORMS.items():
        need = {"private, direct": 8, "private, clean": 10}.get(name, 7)
        assert B.storage_run(False, policy, forms, need - 1, 4, 7, 1000) is None
        r = B.storage_run(False, policy, forms, need, 4, 7, 1000)
        got[name] = (need, r["kinds"], r["per_job_cycles"], r["borrowed"])
    assert got == {
        "private, direct": (8, ["direct"], 14, 0),
        "private, clean": (10, ["clean"], 614, 0),
        "private, spill": (7, ["spill"], 19, 0),
        "lease (static)": (7, ["leased"], 614, 3),
        "least cost (all forms)": (7, ["spill"], 19, 0),
    }

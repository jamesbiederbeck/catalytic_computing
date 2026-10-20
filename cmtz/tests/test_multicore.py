"""Multilane reference model, scheduler and lease invariants (RFC-003 A0)."""

import random

import pytest

from cmtz.analysis.lease_invariants import check_trace
from cmtz.isa import projection as PR
from cmtz.isa.asm import assemble
from cmtz.isa.image import ProgramImage
from cmtz.isa.lease import WINDOW, Lease, relocate, window_conflicts
from cmtz.isa.multicore import (
    E_ABORT, E_BUDGET, Job, LaneConfig, MultiCore, Variant,
)
from cmtz.isa.sim import ERROR, HALTED, Machine
from cmtz.isa import workloads as W

CORE = PR.CORE
P = CORE.p


def solo(image, inputs, R=None, X=None, D=None, irq_at=()):
    m = Machine(CORE, image)
    if R is not None:
        m.R, m.X, m.D = list(R), list(X), list(D)
    m.set_inputs({k: v for k, v in inputs.items() if k in image.inputs})
    m.run(irq_at=irq_at, max_steps=10_000_000)
    return m


def asm_image(src, inputs=None, outputs=None):
    a = assemble(src)
    return ProgramImage(P, CORE.q, a.words, a.entry, a.handler,
                        inputs=inputs or {}, outputs=outputs or {},
                        listing=[str(i) for i in a.instrs])


def mixed_programs(rng):
    """(image, inputs) pairs: the projection programs with random inputs."""
    out = []
    for fn in (PR.gate_direct, PR.gate_projection, PR.gate_clean):
        d = fn()
        ins = {**PR.gate_inputs(rng.randrange(16), rng.randrange(2),
                                rng.randrange(2)),
               **{k: rng.randrange(P) for k in ("y", "t1", "t2", "s")}}
        out.append((d.image, ins))
    d = PR.tree_direct()
    ins = {**PR.tree_inputs(*[rng.randrange(16) for _ in range(3)],
                            *[rng.randrange(2) for _ in range(4)]),
           "y": rng.randrange(P)}
    out.append((d.image, ins))
    out.append((W.owner_program(rng.randrange(1, 300)),
                {"x": rng.randrange(P), "c": rng.randrange(P),
                 **{f"h{i}": rng.randrange(P) for i in range(3)}}))
    out.append((PR.gate_irq(1, 1).image,
                {**PR.gate_inputs(6, 1, 1),
                 **{k: rng.randrange(P) for k in ("y", "t1", "t2", "s")}}))
    return out


class TestIndependentLanes:
    @pytest.mark.parametrize("lanes", [1, 2, 4, 8])
    def test_each_lane_equals_its_own_scalar_run(self, lanes):
        rng = random.Random(lanes)
        progs = mixed_programs(rng)
        mc = MultiCore(CORE, lanes)
        mc.scramble(lanes)
        start_R = [mc.regs(i) for i in range(lanes)]
        start_X = [list(ln.X) for ln in mc.lanes]
        start_D = [list(ln.D) for ln in mc.lanes]
        picks = []
        for i in range(lanes):
            img, ins = progs[i % len(progs)]
            irq = (rng.randrange(40),) if img.handler is not None else ()
            picks.append((img, ins, irq))
            mc.submit(Job(f"j{i}", [Variant(img)], ins, irq_at=irq, lane=i))
        rep = mc.run()
        assert not check_trace(mc.trace)
        for i, (img, ins, irq) in enumerate(picks):
            m = solo(img, ins, start_R[i], start_X[i], start_D[i], irq)
            r = rep.by_name()[f"j{i}"]
            assert (r.status, r.retired, r.outputs) == (m.status, m.retired,
                                                        m.outputs())
            assert r.cycles == m.cycles == r.model_cycles
            assert r.start == 0 and r.stalls == 0
            assert mc.regs(i) == m.R
            assert mc.lanes[i].X == m.X and mc.lanes[i].D == m.D
        assert rep.makespan == max(r.cycles for r in rep.results)
        assert rep.ipc == pytest.approx(
            sum(r.retired for r in rep.results) / rep.makespan)

    def test_queue_back_to_back_on_two_lanes(self):
        rng = random.Random(3)
        jobs = [W.gate_job(f"g{i}", rng.randrange(16), rng.randrange(2),
                           rng.randrange(2), rng.randrange(P), rng,
                           forms="private") for i in range(7)]
        mc = MultiCore(CORE, 2)
        mc.submit(*jobs)
        rep = mc.run()
        assert not rep.pending and not check_trace(mc.trace)
        for j in jobs:
            assert rep.by_name()[j.name].outputs["y"] == j.expect["y"]
        # 7 jobs of 614 cycles on 2 lanes: 4 rounds
        assert rep.makespan == 4 * 614
        assert rep.ipc == pytest.approx(7 * 307 / (4 * 614))

    def test_budget_fault_and_admission(self):
        mc = MultiCore(CORE, 1, [LaneConfig(private=12)])
        rng = random.Random(0)
        job = W.gate_job("g", 6, 1, 0, 3, rng, forms="private")
        assert not mc.fits(0, job.variants[0])      # uses R16
        mc.submit(job)
        rep = mc.run()
        assert rep.pending == ["g"] and not rep.results
        # forced past admission, the lane faults on R16
        mc = MultiCore(CORE, 1, [LaneConfig(private=12)],
                       policy=lambda mc, lane, t, jobs: (jobs[0], 0, None))
        mc.submit(job)
        r = mc.run().results[0]
        assert (r.status, r.error) == (ERROR, E_BUDGET)


class TestStaticLeases:
    def setup_method(self):
        self.lg = W.leased(PR.gate_clean())

    def test_relocated_image_moves_catalytic_registers(self):
        assert self.lg.catalytic == {"t1": 24, "t2": 25, "s": 26}
        assert set(self.lg.image.inputs) == {"a", "b", "k0", "k1", "k2",
                                             "k3", "y"}
        assert not window_conflicts(self.lg.image, 3)
        assert window_conflicts(self.lg.image, 2)     # R26 outside the lease
        # relocating back gives the original program
        back = relocate(self.lg.image, {24: 7, 25: 8, 26: 16})
        assert back.words == PR.gate_clean().image.words

    @pytest.mark.parametrize("seed", range(4))
    def test_leased_gate_restores_owner_registers(self, seed):
        rng = random.Random(seed)
        mc = MultiCore(CORE, 2, policy="static", snapshot=True,
                       trace_access=True)
        mc.scramble(seed, below=P)
        before = mc.regs(0)
        regs = tuple(rng.sample(range(1, 24), 3))
        tt, a, b, y = rng.randrange(16), rng.randrange(2), rng.randrange(2), rng.randrange(P)
        job = W.static_gate("g", 0, regs, tt, a, b, y, lane=1)
        mc.submit(job)
        rep = mc.run()
        r = rep.results[0]
        assert r.outputs["y"] == job.expect["y"]
        assert r.restored is True and r.cycles == 614
        assert mc.regs(0) == before
        assert not check_trace(mc.trace)
        assert rep.snapshot_bits == 3 * CORE.rw
        # the borrower's private state equals a scalar run of the private
        # program with the owner's values in its catalytic registers
        ref = list(mc.regs(1))
        m = Machine(CORE, PR.gate_clean().image)
        R0 = [0] * 32
        mc2 = MultiCore(CORE, 2)
        mc2.scramble(seed, below=P)
        R0 = mc2.regs(1)
        for name, r_ in zip(("t1", "t2", "s"), regs):
            R0[PR.gate_clean().layout.catalytic[name]] = before[r_]
        m.R, m.X, m.D = R0, list(mc2.lanes[1].X), list(mc2.lanes[1].D)
        m.set_inputs({k: v for k, v in job.inputs.items()})
        m.run()
        for r_ in range(32):
            if r_ not in (7, 8, 16):
                assert ref[r_] == m.R[r_], r_

    def test_owner_runs_and_stalls_on_leased_register(self):
        """The owner loops on R1..R3 while lane 1 borrows R16..R18, then
        reads R16..R18 and stalls until the borrower is done."""
        rng = random.Random(1)
        h = (3, 9, 12)
        own = W.owner_job("own", 20, 5, 7, h, lane=0)
        g = W.static_gate("g", 0, W.OWNER_LIVE, 6, 1, 0, 4, lane=1)
        mc = MultiCore(CORE, 2, policy="static", snapshot=True,
                       trace_access=True)
        for r, v in zip(W.OWNER_LIVE, h):
            mc.write_reg(0, r, v)
        mc.submit(g, own)       # lane 1 acquires before lane 0 starts
        rep = mc.run()
        res = rep.by_name()
        assert res["own"].outputs == own.expect
        assert res["g"].outputs["y"] == g.expect["y"]
        solo_own = solo(own.variants[0].image, own.inputs)
        # owner reaches ADD R4, R16, R17 after 20 x 4 + 2 instructions
        stall_at = 2 * (2 + 20 * 4) + 1         # its EXEC cycle
        assert res["own"].stalls == 614 - stall_at
        assert res["own"].cycles == solo_own.cycles + res["own"].stalls
        assert res["g"].stalls == 0 and res["g"].restored
        assert not check_trace(mc.trace)
        assert rep.makespan == 614 + solo_own.cycles - stall_at

    def test_write_to_leased_register_stalls_too(self):
        img = asm_image("LDI R16, 4\nHALT")
        mc = MultiCore(CORE, 2, policy="static", snapshot=True)
        mc.submit(W.static_gate("g", 0, (16, 17, 18), 6, 1, 1, 0, lane=1),
                  Job("w", [Variant(img)], lane=0))
        rep = mc.run()
        assert rep.by_name()["w"].stalls == 614 - 1
        assert mc.regs(0)[16] == 4
        assert not check_trace(mc.trace)

    def test_host_cannot_write_a_leased_register(self):
        mc = MultiCore(CORE, 2)
        assert not mc.acquire(1, Lease(0, (5,)), 0)
        with pytest.raises(Exception):
            mc.write_reg(0, 5, 1)

    def test_lease_refused_for_values_outside_field(self):
        mc = MultiCore(CORE, 2, policy="static")
        mc.P[17] = 18           # R17 of lane 0 holds 18 >= p
        mc.submit(W.static_gate("g", 0, (16, 17, 18), 6, 1, 1, 0, lane=1))
        rep = mc.run()
        assert rep.pending == ["g"] and rep.leases["refuse"] == 1
        assert any(e["ev"] == "refuse" and "outside" in e["why"][0]
                   for e in mc.trace)

    def test_lease_shape_limits(self):
        with pytest.raises(ValueError):
            Lease(0, tuple(range(1, 10)))           # nine registers
        with pytest.raises(ValueError):
            Lease(0, (3, 3))
        with pytest.raises(ValueError):
            Lease(0, (0, 1))
        assert Lease.banked(1, 2, 3).regs == (16, 17, 18)
        assert Lease.banked(1, 2, 3).bank == (2, 3)
        assert Lease(1, (16, 18)).bank is None
        with pytest.raises(ValueError):
            Lease.banked(0, 0, 2)

    def test_no_lane_lends_and_borrows(self):
        mc = MultiCore(CORE, 3)
        assert not mc.acquire(1, Lease(0, (5,)), 0)
        # lane 0 is lending, so it may not borrow; lane 1 borrowing may
        # not lend
        assert mc.acquire(0, Lease(2, (5,)), 0)
        assert mc.acquire(2, Lease(1, (6,)), 0)
        # the cycle 0 -> 1 -> 0 is the deadlock this rule prevents
        assert mc.acquire(0, Lease(1, (6,)), 0)
        assert mc.acquire(1, Lease(2, (7,)), 0)     # second lease
        assert mc.acquire(2, Lease(0, (5,)), 0)     # register taken

    def test_library_image_cannot_borrow(self):
        from cmtz.backends.isa_backend import emit_isa
        from cmtz.compiler import compile_program
        from pathlib import Path
        src = (Path(__file__).resolve().parents[2] / "examples"
               / "fibonacci.cmtz").read_text()
        b = emit_isa(compile_program(src, backend="python").ir_program, CORE,
                     matrix_dims={"fib_matrix": 2})
        errs = window_conflicts(b.image, 3)
        assert errs and "library" in errs[0]
        mc = MultiCore(CORE, 2, policy="static")
        mc.submit(Job("lib", [Variant(b.image, "leased",
                                      lease=Lease(0, (16, 17, 18)))], lane=1))
        rep = mc.run()
        assert rep.pending == ["lib"] and rep.leases["refuse"] == 1


class TestPoison:
    def _faulting_borrower(self):
        # writes into the window, then RET with an empty stack
        return asm_image("ADD R24, R24, R1\nRET\nHALT", inputs={"v": 1})

    def test_borrower_error_poisons_and_owner_is_aborted(self):
        img = self._faulting_borrower()
        mc = MultiCore(CORE, 2, policy="static", snapshot=True)
        for r, v in zip((16, 17, 18), (1, 2, 3)):
            mc.write_reg(0, r, v)
        reader = asm_image("ADD R5, R16, R16\nHALT", outputs={"o": 5})
        mc.submit(Job("bad", [Variant(img, "leased",
                                      lease=Lease(0, (16, 17, 18)))],
                      {"v": 5}, lane=1),
                  Job("rd", [Variant(reader)], lane=0, after=("bad",)))
        rep = mc.run()
        res = rep.by_name()
        assert res["bad"].status == ERROR
        assert rep.leases["poison"] == 1 and rep.leases["release"] == 0
        assert (res["rd"].status, res["rd"].error) == (ERROR, E_ABORT)
        assert mc.regs(0)[16] == 6            # the borrower's half-done write
        assert {16, 17, 18} <= {i % 32 for i in mc.poisoned}
        assert not check_trace(mc.trace)
        # host recovery: restore the value and unlock
        mc.clear_poison(0, {16: 1, 17: 2, 18: 3})
        mc.submit(Job("rd2", [Variant(reader)], lane=0))
        rep = mc.run()
        assert rep.by_name()["rd2"].outputs == {"o": 2}
        assert not check_trace(mc.trace)

    def test_poisoned_registers_cannot_be_leased_again(self):
        mc = MultiCore(CORE, 2, policy="static")
        mc.submit(Job("bad", [Variant(self._faulting_borrower(), "leased",
                                      lease=Lease(0, (16,)))], {"v": 1},
                      lane=1))
        mc.run()
        assert mc.acquire(1, Lease(0, (16,)), 0)


class TestInvariantChecker:
    """The checker must catch what it claims to: feed it broken traces."""

    def _good(self):
        mc = MultiCore(CORE, 2, policy="static", snapshot=True,
                       trace_access=True)
        mc.submit(W.static_gate("g", 0, (16, 17, 18), 6, 1, 1, 0, lane=1),
                  W.owner_job("own", 5, 1, 2, (0, 0, 0), lane=0))
        mc.run()
        assert not check_trace(mc.trace)
        return mc.trace

    def test_detects_corruptions(self):
        t = self._good()
        rel = next(i for i, e in enumerate(t) if e["ev"] == "release")
        acc = next(i for i, e in enumerate(t) if e["ev"] == "access"
                   and e["lane"] == 0)
        cases = {
            "values": lambda tr: tr[rel].update(values=[1, 2, 3]),
            "early release": lambda tr: tr[rel].update(t=tr[rel]["t"] - 1),
            "double lease": lambda tr: tr.insert(1, dict(tr[0], lane=0,
                                                         owner=1,
                                                         phys=[16 + 32])),
            "owner touches": lambda tr: tr.insert(acc + 1, dict(
                tr[acc], phys=[16])) if tr[acc]["t"] < tr[rel]["t"] else None,
            "no release": lambda tr: tr.pop(rel),
            "stall sum": lambda tr: next(e for e in tr if e["ev"] == "end"
                                         and e["lane"] == 0).update(stalls=1),
        }
        for name, corrupt in cases.items():
            tr = [dict(e) for e in t]
            corrupt(tr)
            assert check_trace(tr), name


class TestRandomSchedules:
    def _jobs(self, rng, n):
        jobs = []
        for i in range(n):
            k = rng.random()
            if k < 0.6:
                jobs.append(W.gate_job(f"g{i}", rng.randrange(16),
                                       rng.randrange(2), rng.randrange(2),
                                       rng.randrange(P), rng))
            elif k < 0.7:
                bits = tuple(rng.randrange(2) for _ in range(4))
                jobs.append(W.tree_job(f"t{i}", *[rng.randrange(16) for _ in range(3)],
                                       bits, rng.randrange(P), rng))
            else:
                jobs.append(W.owner_job(
                    f"o{i}", rng.randrange(1, 60), rng.randrange(P),
                    rng.randrange(P), tuple(rng.randrange(P) for _ in range(3)),
                    lane=None))
            jobs[-1].arrival = rng.randrange(3000)
        return jobs

    @pytest.mark.parametrize("seed", range(12))
    def test_dynamic_random_runs_keep_every_invariant(self, seed):
        rng = random.Random(seed)
        lanes = rng.choice([2, 3, 4])
        cfgs = [LaneConfig(lendable=(19, 20, 21, 22, 23)) for _ in range(lanes)]
        mc = MultiCore(CORE, lanes, cfgs, policy="dynamic", seed=seed,
                       randomize=True, jitter=5, snapshot=True,
                       trace_access=seed < 3)
        mc.scramble(seed, below=P)
        jobs = self._jobs(rng, 14)
        mc.submit(*jobs)
        rep = mc.run()
        assert not rep.pending
        assert not check_trace(mc.trace)
        for j in jobs:
            r = rep.by_name()[j.name]
            assert r.status == HALTED, j.name
            assert r.outputs.get("y", r.outputs.get("r")) == \
                j.expect.get("y", j.expect.get("r")), j.name
            if r.lease:
                assert r.restored is True
        assert rep.leases["acquire"] == rep.leases["release"] > 0

    def test_replay_reproduces_the_run(self):
        rng = random.Random(7)
        cfgs = [LaneConfig(lendable=(19, 20, 21, 22, 23)) for _ in range(3)]

        def make(**kw):
            mc = MultiCore(CORE, 3, cfgs, policy="dynamic", snapshot=True, **kw)
            mc.scramble(7, below=P)
            mc.submit(*self._jobs(random.Random(7), 10))
            return mc
        a = make(seed=11, randomize=True, jitter=9)
        ra = a.run()
        b = make(seed=11, randomize=True, jitter=9)
        assert b.run().trace_hash == ra.trace_hash          # same seed
        c = make(seed=99, replay=a.decisions)               # no generator
        rc = c.run()
        assert rc.trace_hash == ra.trace_hash
        d = make(seed=12, randomize=True, jitter=9)
        assert d.run().trace_hash != ra.trace_hash

    def test_policies_on_one_workload(self):
        """Same jobs under private, static and dynamic: same outputs."""
        rng = random.Random(5)
        outs = {}
        for pol in ("private", "dynamic"):
            mc = MultiCore(CORE, 2, [LaneConfig(lendable=(19, 20, 21))] * 2,
                           policy=pol, snapshot=True)
            mc.scramble(1, below=P)
            jobs = [W.gate_job(f"g{i}", i % 16, i % 2, (i // 2) % 2, i % P,
                               random.Random(i)) for i in range(6)]
            mc.submit(*jobs)
            rep = mc.run()
            assert not check_trace(mc.trace)
            outs[pol] = {r.name: r.outputs["y"] for r in rep.results}
            kinds = {r.kind for r in rep.results}
            assert kinds == ({"private"} if pol == "private" else
                             {"leased", "private"})
        assert outs["private"] == outs["dynamic"]

"""Proof-aware leases (RFC-003 A3): certificates on the allocated machine
code, refusal of everything the proof does not cover, scheduler use."""

import random
from dataclasses import replace

import pytest

from cmtz.analysis.isa_clean import Poly
from cmtz.analysis.lease_invariants import check_trace
from cmtz.analysis.lease_proof import (
    Checker, Claim, LeaseRefused, certify, check_lease,
)
from cmtz.isa import projection as PR
from cmtz.isa import workloads as W
from cmtz.isa.asm import assemble
from cmtz.isa.image import ProgramImage
from cmtz.isa.lease import Lease
from cmtz.isa.multicore import Job, LaneConfig, MultiCore, Variant
from cmtz.isa.sim import HALTED

CORE = PR.CORE
P = CORE.p
A, B, K, Y = PR.A, PR.B, PR.K, PR.Y
GATE_IN = {"a": A, "b": B, "k0": 3, "k1": 4, "k2": 5, "k3": 6, "y": Y}


def gate_src(t1=24, t2=25, s=26, extra=""):
    return f"""
        .entry main
main:   LIX   X3, {PR.M}{PR._clean_loop("loop", A, B, K, Y, t1, t2, s, 1, 3)}
        {extra}
        HALT
"""


def image(src, inputs=GATE_IN, outputs=None):
    a = assemble(src)
    return ProgramImage(P, CORE.q, a.words, a.entry, a.handler,
                        inputs=dict(inputs), outputs=outputs or {"y": Y},
                        listing=[str(i) for i in a.instrs])


GAIN = Claim.of({Y: PR.expected_gain(GATE_IN)})


def refused(img, window=3, claim=GAIN) -> list[str]:
    with pytest.raises(LeaseRefused) as e:
        certify(img, window, claim)
    return e.value.violations


class TestCertify:
    def test_leased_programs_certify(self):
        for demo, n in ((PR.gate_clean(), 3), (PR.tree_clean(), 5)):
            lg = W.leased(demo)
            c = certify(lg.image, n, lg.claim)
            assert c.image_crc == lg.image.crc and c.window == n
            assert set(c.catalytic) <= set(c.requires_canonical)
            assert c.retired == PR.prove(demo).retired
        assert certify(image(gate_src()), 3, GAIN).retired == 307

    def test_non_restoring_program(self):
        # 15 rotations instead of 16: the masks come back as w^15 t
        src = gate_src().replace(f"LIX   X3, {PR.M}", "LIX   X3, 15", 1)
        errs = refused(image(src))
        assert any("catalytic R24 ends" in e for e in errs)

    def test_missing_uncompute_gives_the_wrong_gain(self):
        # the masks still come back (sum of w^k is 0) but y gains g at 8a
        src = gate_src().replace("SUB   R24, R24, R1", "NOP", 1)
        errs = refused(image(src))
        assert any("output R9 gains" in e for e in errs)
        assert not any("catalytic" in e for e in errs)

    def test_rotation_root_orders(self):
        # any root of order > 2 cancels the degree-2 cross terms over 16
        # steps (w^2 = 9, order 8; 4, order 4): accepted. -1 (order 2)
        # leaves the t1 t2 term: refused.
        for k, ok in ((9, True), (4, True), (16, False)):
            src = (gate_src().replace("MULI  R24, R24, 3", f"MULI  R24, R24, {k}", 1)
                   .replace("MULI  R25, R25, 3", f"MULI  R25, R25, {k}", 1))
            if ok:
                assert certify(image(src), 3, GAIN)
            else:
                assert refused(image(src))

    def test_wrong_claim(self):
        src = gate_src()
        bad = Claim.of({Y: PR.expected_gain(GATE_IN) + Poly.const(P, 1)})
        assert any("output R9 gains" in e for e in refused(image(src), claim=bad))

    def test_undeclared_scratch(self):
        src = gate_src(extra="LDI R12, 3")
        assert any("R12" in e for e in refused(image(src)))
        ok = Claim.of({Y: PR.expected_gain(GATE_IN)}, temps=(12,))
        assert certify(image(src), 3, ok)

    def test_ownership(self):
        # touches R27 with a three-register window
        src = gate_src(extra="ADD R27, R27, R0")
        assert any("outside the lease" in e for e in refused(image(src)))
        # the host writes into the window
        assert any("host-written" in e for e in refused(
            image(gate_src(), inputs={**GATE_IN, "t1": 24})))
        # a window bigger than the program's use is fine; smaller is not
        assert refused(image(gate_src()), window=2)

    def test_library_program(self):
        from pathlib import Path
        from cmtz.backends.isa_backend import emit_isa
        from cmtz.compiler import compile_program
        src = (Path(__file__).resolve().parents[2] / "examples"
               / "fibonacci.cmtz").read_text()
        b = emit_isa(compile_program(src, backend="python").ir_program, CORE,
                     matrix_dims={"fib_matrix": 2})
        assert refused(b.image, claim=Claim.of({}))

    def test_unsupported_memory_and_control(self):
        assert any("LD" in e for e in refused(image(gate_src(extra="LD R12, [X0+0]"))))
        assert any("ST" in e for e in refused(image(gate_src(extra="ST R24, [X0+0]"))))
        branchy = gate_src(extra="BZ R24, done\nNOP\ndone: NOP")
        assert any("depends on the inputs" in e for e in refused(image(branchy)))

    def test_entry_value_nobody_supplies(self):
        src = gate_src(extra="ADD R9, R9, R13\nSUB R9, R9, R13")
        assert any("R13" in e for e in refused(image(src)))

    def test_handler_on_the_window(self):
        src = (".handler isr\n" + gate_src()
               + "isr: ADD R25, R25, R1\nSUB R25, R25, R1\nRETI\n")
        img = replace(image(src), borrowed=[25], handler_scratch=[])
        assert any("handler names window" in e for e in refused(img))

    def test_verified_handler_off_the_window(self):
        src = (".handler isr\n" + gate_src()
               + "isr: LDI R14, 3\nMUL R15, R14, R14\nRETI\n")
        img = replace(image(src), handler_scratch=[14, 15])
        c = certify(img, 3, GAIN)
        assert c.handler.startswith("verified")


class TestBinding:
    def setup_method(self):
        self.lg = W.leased(PR.gate_clean())
        self.cert = certify(self.lg.image, 3, self.lg.claim)

    def test_matching_lease(self):
        assert not check_lease(self.cert, self.lg.image, Lease(0, (16, 17, 18)), 1)

    def test_other_program(self):
        other = W.leased(PR.tree_clean()).image
        assert check_lease(self.cert, other, Lease(0, (16, 17, 18)), 1)

    def test_wrong_size_and_aliasing(self):
        assert check_lease(self.cert, self.lg.image, Lease(0, (16, 17)), 1)
        assert check_lease(self.cert, self.lg.image, Lease(1, (16, 17, 18)), 1)

    def test_owner_handler_on_leased_register(self):
        own = PR.gate_irq(1, 1).image          # its handler uses R7, R14, R15
        assert check_lease(self.cert, self.lg.image, Lease(0, (7, 8, 9)), 1, own)
        assert not check_lease(self.cert, self.lg.image, Lease(0, (16, 17, 18)),
                               1, own)


class TestScheduler:
    def test_only_proven_programs_borrow(self):
        """Two leased variants, one restoring and one not: the checker
        accepts the first, refuses the second, which then runs privately."""
        rng = random.Random(0)
        good = W.gate_job("good", 6, 1, 0, 3, rng)
        bad_img = image(gate_src().replace("SUB   R24, R24, R1", "NOP", 1))
        bad = W.gate_job("bad", 6, 1, 0, 3, rng)
        bad.variants[0] = replace(bad.variants[0], image=bad_img)
        chk = Checker()
        mc = MultiCore(CORE, 2, [LaneConfig(lendable=(16, 17, 18))] * 2,
                       policy="dynamic", snapshot=True, accept=chk)
        mc.scramble(1, below=P)
        mc.submit(good, bad)
        rep = mc.run()
        res = rep.by_name()
        assert res["good"].kind == "leased" and res["good"].restored
        assert res["bad"].kind == "private"
        assert res["bad"].outputs["y"] == bad.expect["y"]
        assert chk.proofs == 2 and chk.refused
        acq = [e for e in mc.trace if e["ev"] == "acquire"]
        assert len(acq) == 1 and acq[0]["cert"]
        assert not check_trace(mc.trace)

    def test_no_claim_no_lease(self):
        lg = W.leased(PR.gate_clean())
        job = Job("g", [Variant(lg.image, "leased", window=3)],
                  PR.gate_inputs(6, 1, 0) | {"y": 0})
        mc = MultiCore(CORE, 2, [LaneConfig(lendable=(16, 17, 18))] * 2,
                       policy="dynamic", accept=Checker())
        mc.submit(job)
        rep = mc.run()
        assert rep.pending == ["g"] and rep.leases["acquire"] == 0

    def test_random_dynamic_runs_with_proofs(self):
        rng = random.Random(4)
        chk = Checker()
        mc = MultiCore(CORE, 3, [LaneConfig(lendable=(19, 20, 21, 22, 23))] * 3,
                       policy="dynamic", seed=4, randomize=True, snapshot=True,
                       accept=chk)
        mc.scramble(4, below=P)
        jobs = [W.gate_job(f"g{i}", rng.randrange(16), rng.randrange(2),
                           rng.randrange(2), rng.randrange(P), rng)
                for i in range(10)]
        jobs += [W.tree_job("t", 6, 8, 14, (1, 0, 1, 1), 5, rng)]
        mc.submit(*jobs)
        rep = mc.run()
        assert not check_trace(mc.trace)
        for j in jobs:
            r = rep.by_name()[j.name]
            assert r.status == HALTED and r.outputs["y"] == j.expect["y"]
        assert rep.leases["acquire"] > 3 and rep.leases["not_restored"] == 0
        assert chk.proofs <= 2 and chk.hits > 0  # one per program, then cached
        assert all("cert" in e for e in mc.trace if e["ev"] == "acquire")

"""Projection programs on the cmtz-core simulator, against the oracle."""

import itertools
import random

import pytest

from cmtz.analysis.isa_catalytic import IsaCatalyticError, verify_handler
from cmtz.isa import Machine, SimError
from cmtz.isa import projection as PR
from cmtz.isa.sim import HALTED, run_image
from cmtz.projection_oracle import gate, gate_tree, masked_gate

CORE = PR.CORE
P = CORE.p
BITS = (0, 1)


def _run(demo, values, seed):
    """Run from scrambled leftovers; return (machine before, after)."""
    m = Machine(CORE, demo.image)
    m.scramble(seed)
    m.set_inputs(values)
    before = Machine(CORE, demo.image)
    before.R, before.X = list(m.R), list(m.X)
    m.run()
    assert m.status == HALTED
    return before, m


def _check_clean(demo, before, after, gained):
    """Inputs untouched, catalytic registers back, outputs gained `gained`;
    only temporaries and index workspace may differ."""
    lay = demo.layout
    free = set(lay.temps) | set(lay.outputs.values())
    for r in range(32):
        if r not in free:
            assert after.R[r] == before.R[r], f"R{r}"
    for name, r in lay.outputs.items():
        assert after.R[r] == (before.R[r] + gained[name]) % P, name
    for x in range(16):
        if x not in lay.index:
            assert after.X[x] == before.X[x], f"X{x}"


def test_layouts_are_consistent():
    for fn in (PR.gate_direct, PR.gate_projection, PR.gate_clean,
               PR.gate_sweep, PR.tree_direct, PR.tree_clean):
        d = fn()
        d.image.validate(CORE)
        lay = d.layout
        roles = [set(lay.inputs.values()), set(lay.catalytic.values()),
                 set(lay.outputs.values()), set(lay.temps)]
        for x, y in itertools.combinations(roles, 2):
            assert not (x & y), fn.__name__
        assert 0 not in set().union(*roles)


@pytest.mark.parametrize("fn", [PR.gate_direct, PR.gate_projection, PR.gate_clean])
def test_gate_programs_match_oracle(fn):
    """All 16 gates x 4 inputs, three random mask/accumulator draws each,
    from scrambled leftovers."""
    demo = fn()
    rng = random.Random(fn.__name__)
    for tt, a, b in itertools.product(range(16), BITS, BITS):
        for k in range(3):
            t1, t2, sv, y = (rng.randrange(P) for _ in range(4))
            vals = {**PR.gate_inputs(tt, a, b), "y": y}
            if "t1" in demo.image.inputs:
                vals.update(t1=t1, t2=t2)
            if "s" in demo.image.inputs:
                vals.update(s=sv)
            before, after = _run(demo, vals, seed=tt * 16 + 4 * a + 2 * b + k)
            _check_clean(demo, before, after, {"y": gate(tt, a, b)})
            if fn is PR.gate_clean:
                assert (after.R[PR.Y], after.R[PR.T1], after.R[PR.T2]) == \
                    masked_gate(tt, a, b, t1, t2, y)


def test_gate_clean_exhaustive_over_masks():
    """gate_sweep runs gate_clean for every (T1, T2) in F_17^2; one run per
    gate and input, the starting accumulator cycling through F_17."""
    demo = PR.gate_sweep()
    for n, (tt, a, b) in enumerate(itertools.product(range(16), BITS, BITS)):
        y0 = n % P
        m = run_image(CORE, demo.image,
                      {**PR.gate_inputs(tt, a, b), "y0": y0, "s": (5 * n) % P},
                      scramble=n)
        assert m.status == HALTED
        assert m.D[:3 * P * P] == PR.sweep_expected(tt, a, b, y0), (tt, a, b)


def test_tree_programs_match_oracle():
    clean, direct = PR.tree_clean(), PR.tree_direct()
    rng = random.Random(3)
    triples = [(rng.randrange(16), rng.randrange(16), rng.randrange(16))
               for _ in range(8)]
    for (f, g1, g2), (a, b, c, d) in itertools.product(
            triples, itertools.product(BITS, repeat=4)):
        regs = {k: rng.randrange(P) for k in ("t1", "t2", "tl", "tr", "s", "y")}
        vals = PR.tree_inputs(f, g1, g2, a, b, c, d)
        want = gate(f, gate(g1, a, b), gate(g2, c, d))
        seed = rng.randrange(1 << 30)
        before, after = _run(clean, {**vals, **regs}, seed)
        _check_clean(clean, before, after, {"y": want})
        assert {k: after.R[r] for k, r in clean.image.outputs.items()} == \
            gate_tree(f, g1, g2, a, b, c, d, regs)
        before, after = _run(direct, {**vals, "y": regs["y"]}, seed)
        _check_clean(direct, before, after, {"y": want})


def test_retired_counts_are_data_independent():
    """No branch depends on a field value, so each program retires the same
    number of instructions for every input."""
    counts = {}
    for fn in (PR.gate_direct, PR.gate_projection, PR.gate_clean,
               PR.tree_direct, PR.tree_clean):
        demo, seen = fn(), set()
        for seed in range(4):
            rng = random.Random(seed)
            vals = {k: rng.randrange(P) for k in demo.image.inputs}
            seen.add(run_image(CORE, demo.image, vals, scramble=seed).retired)
        assert len(seen) == 1, fn.__name__
        counts[fn.__name__] = seen.pop()
    assert counts == {"gate_direct": 7, "gate_projection": 212,
                      "gate_clean": 307, "tree_direct": 17,
                      "tree_clean": 19955}


class TestInterrupt:
    def test_handler_verifies(self):
        assert verify_handler(PR.gate_irq(1, 1).image).borrowed == [PR.T1]
        with pytest.raises(IsaCatalyticError):
            verify_handler(PR.gate_irq(1, 1, restore=False).image)

    def test_every_dispatch_point(self):
        """The handler borrows mask t1 while gate_clean is using it. At every
        instruction boundary, the run ends exactly as without it."""
        rng = random.Random(5)
        good, bad = PR.gate_irq(1, 1), PR.gate_irq(1, 1, restore=False)
        changed = 0
        for tt in range(16):
            a, b = rng.randrange(2), rng.randrange(2)
            vals = {**PR.gate_inputs(tt, a, b),
                    **{k: rng.randrange(P) for k in ("t1", "t2", "s", "y")}}
            ref = run_image(CORE, good.image, vals, irq_at=())
            assert ref.waiting is False and ref.status == HALTED
            for k in range(ref.retired):
                m = run_image(CORE, good.image, vals, irq_at=(k,))
                assert m.dispatches and m.dispatches[0][0] == k
                assert m.outputs() == ref.outputs(), (tt, k)
                assert m.retired == ref.retired + 7
                mb = run_image(CORE, bad.image, vals, irq_at=(k,))
                changed += mb.outputs() != ref.outputs()
        assert changed > 0


def test_mask_outside_the_field_is_refused_by_the_simulator():
    """The register can hold 17 or 18 (q = 19); the simulator refuses to
    guess, since the hardware would reduce it and not restore it."""
    demo = PR.gate_clean()
    m = Machine(CORE, demo.image)
    m.set_inputs({**PR.gate_inputs(8, 1, 1), "t1": 3, "t2": 4, "y": 0})
    m.R[PR.T1] = 18
    with pytest.raises(SimError):
        m.run()


def test_benchmark_numbers():
    """Pinned so a change to the programs shows up in review; the docs quote
    them. Cycles are 2 per instruction (no LD in these programs)."""
    got = {r["program"]: (r["words"], r["instructions"], r["cycles"],
                          r["catalytic"], r["temps"], r["field_regs"])
           for r in PR.benchmark()}
    assert got == {
        "gate_direct": (7, 7, 14, 0, 1, 8),
        "gate_projection": (17, 212, 424, 0, 4, 13),
        "gate_clean": (22, 307, 614, 3, 0, 10),
        "tree_direct": (17, 17, 34, 0, 3, 20),
        "tree_clean": (68, 19955, 39910, 5, 0, 22),
    }

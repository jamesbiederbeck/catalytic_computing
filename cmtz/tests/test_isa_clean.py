"""Clean-computation proof on cmtz-core machine code (analysis/isa_clean.py)."""

import itertools

import pytest

from cmtz.analysis.isa_clean import CleanProofError, Poly, run_symbolic, verify_clean
from cmtz.isa import projection as PR
from cmtz.isa.sim import run_image
from cmtz.projection_oracle import gate

P = 17
PROGRAMS = [PR.gate_direct, PR.gate_projection, PR.gate_clean,
            PR.tree_direct, PR.tree_clean]


def test_poly_reduces_with_fermat():
    x = Poly.var(P, 1)
    x17 = x
    for _ in range(16):
        x17 = x17 * x
    assert x17 == x                         # x^17 = x on F_17
    assert x * x != x
    assert (x + x).scale(9) == x            # 18 = 1
    assert Poly.const(P, 17) == Poly(P)
    assert (x * Poly.var(P, 2)).evaluate({1: 3, 2: 5}) == 15


@pytest.mark.parametrize("fn", PROGRAMS)
def test_programs_are_proven(fn):
    """Each program's output gains ĝ (or f̂(ĝ1, ĝ2)) as a polynomial in
    the input and coefficient registers, catalytic registers come back,
    nothing else changes; the proof's path matches the simulator's."""
    demo = fn()
    rep = PR.prove(demo)
    assert rep.catalytic == sorted(demo.layout.catalytic.values())
    m = run_image(PR.CORE, demo.image, {k: 1 for k in demo.image.inputs})
    assert (rep.retired, rep.cycles) == (m.retired, m.cycles)


def test_proven_tree_gain_is_the_boolean_tree():
    """The proven gain, evaluated at every gate triple's coefficients and
    every Boolean input, is the tree's truth value."""
    demo = PR.tree_clean()
    run = run_symbolic(demo.image)
    y = demo.layout.outputs["y"]
    gain = run.R[y] - Poly.var(P, y)
    ins = demo.layout.inputs
    for f, g1, g2 in itertools.product(range(16), repeat=3):
        for a, b, c, d in itertools.product((0, 1), repeat=4):
            vals = PR.tree_inputs(f, g1, g2, a, b, c, d)
            got = gain.evaluate({ins[k]: v for k, v in vals.items()})
            assert got == gate(f, gate(g1, a, b), gate(g2, c, d))


def _mutant(src_old, src_new, fn=PR.gate_clean):
    demo = fn()
    assert src_old in demo.source
    return PR._image(demo.source.replace(src_old, src_new, 1), demo.layout)


@pytest.mark.parametrize("old,new,why", [
    # no uncompute: t1 still comes back (Σ ω^k = 0), but the arguments are
    # shifted, so the gain is ĝ at a/(1 - ω) = 8a (as the oracle predicts)
    ("SUB   R7, R7, R1", "NOP", r"gains R3 \+ 8\*R1\*R4"),
    ("MULI  R7, R7, 3", "NOP", "output R9 gains"),        # no rotation
    ("LIX   X3, 16", "LIX   X3, 15", "output R9 gains"),  # 15 rotations
    ("MULI  R9, R9, 16       ; -acc", "NOP", "output R9 gains"),   # sign
    ("        HALT", "        ADD   R16, R16, R1\n        HALT", "catalytic R16"),
    ("        HALT", "        MULI  R8, R8, 3\n        HALT", "catalytic R8"),
])
def test_broken_programs_are_rejected(old, new, why):
    with pytest.raises(CleanProofError, match=why):
        PR.prove(_mutant(old, new))


def test_rotation_exponents_are_what_matter():
    """Rotating t2 by 5 = ω^5 instead of ω: a monomial t1^i t2^l picks up
    ω^(j(i + 5l)), and i + 5l is never 0 mod 16 for 0 < i + l <= 2, so the
    proof still goes through. By 6 = ω^15, t1·t2 gets ω^(16j) = 1 and
    survives the sum: the gain depends on the masks."""
    assert pow(3, 5, P) == 5 and pow(3, 15, P) == 6
    PR.prove(_mutant("MULI  R8, R8, 3", "MULI  R8, R8, 5"))
    with pytest.raises(CleanProofError) as e:
        PR.prove(_mutant("MULI  R8, R8, 3", "MULI  R8, R8, 6"))
    assert any("R9 gains" in v and "R7" in v for v in e.value.violations)


def test_alternating_scratch_sign_cancels():
    """Without the last negation s alternates between σ and -σ, so the
    accumulator picks up -2·(±σ)·v each round. (-1)^j = ω^(8j), and
    Σ_j ω^(8j)·(b + ω^j τ2) = 0, so the result is still exact: the proof
    accepts it, correctly."""
    PR.prove(_mutant("MULI  R16, R16, 16         ; s = sigma", "NOP"))


def test_undeclared_change_is_rejected():
    demo = PR.gate_clean()
    lay = demo.layout
    with pytest.raises(CleanProofError, match="R9 is not declared"):
        verify_clean(demo.image, list(lay.catalytic.values()), {}, [])


def test_reading_a_leftover_is_rejected():
    """R25 is not host-written: even though it cancels, its value may be
    17 or 18 on the board."""
    demo = _mutant("        HALT", "        ADD   R9, R9, R25\n"
                   "        SUB   R9, R9, R25\n        HALT")
    with pytest.raises(CleanProofError, match="R25, which the host does not write"):
        PR.prove(demo)


@pytest.mark.parametrize("code,why", [
    ("BZ R1, end", "depends on the inputs"),
    ("HORNER R10, R1, 3", "not supported"),
    ("LIX X1, 0\nLD R10, [X1]", "not supported"),
    ("BLTX X5, X6, end", "X5 is not known"),
])
def test_unprovable_programs_are_refused(code, why):
    demo = PR.gate_clean()
    src = demo.source.replace("main:   LIX   X3, 16",
                              f"main:   {code}\n        LIX   X3, 16")
    src = src.replace("        HALT\n", "end:    HALT\n")
    with pytest.raises(CleanProofError, match=why):
        PR.prove(PR._image(src, demo.layout))

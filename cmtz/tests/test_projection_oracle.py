"""The roots-of-unity projection identity, checked on the plain-integer
reference model (`cmtz/projection_oracle.py`) before any machine code."""

import itertools
import random

import pytest

from cmtz.projection_oracle import (
    GATES, OMEGA, P, coeffs, extension, gate, gate_tree, masked_gate, order,
    projection_sum,
)

BITS = (0, 1)
F = range(P)


def test_omega_has_order_16():
    assert order(OMEGA) == 16 == P - 1
    assert 16 % P == P - 1          # m = -1 in F_17, so 1/m = -1


def test_gate_names_match_their_truth_tables():
    ops = {
        "FALSE": lambda a, b: 0, "TRUE": lambda a, b: 1,
        "AND": lambda a, b: a & b, "OR": lambda a, b: a | b,
        "XOR": lambda a, b: a ^ b, "NAND": lambda a, b: 1 - (a & b),
        "NOR": lambda a, b: 1 - (a | b), "XNOR": lambda a, b: 1 - (a ^ b),
        "A": lambda a, b: a, "B": lambda a, b: b,
        "NOT_A": lambda a, b: 1 - a, "NOT_B": lambda a, b: 1 - b,
        "A_AND_NOT_B": lambda a, b: a & (1 - b),
        "B_AND_NOT_A": lambda a, b: b & (1 - a),
        "A_IMPLIES_B": lambda a, b: (1 - a) | b,
        "B_IMPLIES_A": lambda a, b: (1 - b) | a,
    }
    assert sorted(GATES.values()) == list(range(16))
    for name, tt in GATES.items():
        for a, b in itertools.product(BITS, BITS):
            assert gate(tt, a, b) == ops[name](a, b), (name, a, b)


@pytest.mark.parametrize("tt", range(16))
def test_extension_agrees_on_booleans(tt):
    for a, b in itertools.product(BITS, BITS):
        assert extension(tt, a, b) == gate(tt, a, b)
    assert all(0 <= k < P for k in coeffs(tt))


@pytest.mark.parametrize("mono", ["1", "u", "v", "uv"])
def test_projection_sum_is_m_times_value(mono):
    """Σ_j P(x + ω^j τ1, y + ω^j τ2) = 16·P(x, y) for the monomials every
    ĝ is built from, at every x, y, τ1 in F_17 (all of F_17, not only
    Booleans) and every τ2 for the product term."""
    poly = {"1": lambda u, v: 1, "u": lambda u, v: u, "v": lambda u, v: v,
            "uv": lambda u, v: u * v % P}[mono]
    t2s = F if mono == "uv" else (0, 1, 16)
    for x, y, t1, t2 in itertools.product(F, F, F, t2s):
        assert projection_sum(poly, x, y, t1, t2) == 16 * poly(x, y) % P


@pytest.mark.parametrize("tt", range(16))
def test_masked_gate_exhaustive(tt):
    """Every gate, Boolean input, pair of masks and starting accumulator:
    the accumulator gains g(a, b) and both masks come back."""
    for a, b, t1, t2, y in itertools.product(BITS, BITS, F, F, F):
        assert masked_gate(tt, a, b, t1, t2, y) == ((y + gate(tt, a, b)) % P, t1, t2)


def test_degree_m_breaks_the_identity():
    """Degree 16 = m: the z^16 term survives the sum, so the result depends
    on the masks."""
    poly = lambda u, v: pow(u, 16, P)
    bad = [(x, t) for x in F for t in F
           if projection_sum(poly, x, 0, t, 0) != 16 * poly(x, 0) % P]
    assert bad and all(t != 0 for _, t in bad)
    # total degree 15 still projects exactly
    poly15 = lambda u, v: pow(u, 14, P) * v % P
    assert all(projection_sum(poly15, x, y, t1, t2) == 16 * poly15(x, y) % P
               for x, y, t1, t2 in itertools.product(F, BITS, F, (0, 5)))


def test_wrong_root_order_breaks_the_identity():
    """ω = 4 has order 4: degree < 4 projects (with m = 4), degree 4 does
    not."""
    assert order(4) == 4
    cube = lambda u, v: pow(u, 3, P)
    quart = lambda u, v: pow(u, 4, P)
    for x, t in itertools.product(F, F):
        assert projection_sum(cube, x, 0, t, 0, omega=4) == 4 * cube(x, 0) % P
    assert any(projection_sum(quart, x, 0, t, 0, omega=4) != 4 * quart(x, 0) % P
               for x, t in itertools.product(F, F))


@pytest.mark.parametrize("skip", ["uncompute", "rotate"])
def test_skipping_a_step_breaks_the_gate(skip):
    """Controls. Without the rotation the gain depends on the masks. Without
    the uncompute the arguments are still ω^j·τ' + const, so the gain is
    mask-independent but wrong: ĝ at a/(1 - ω) instead of a."""
    def variant(tt, a, b, t1, t2, y):
        for _ in range(16):
            t1, t2 = (t1 + a) % P, (t2 + b) % P
            y = (y - extension(tt, t1, t2)) % P
            if skip != "uncompute":
                t1, t2 = (t1 - a) % P, (t2 - b) % P
            if skip != "rotate":
                t1, t2 = t1 * OMEGA % P, t2 * OMEGA % P
        return y
    gains = {variant(GATES["AND"], 1, 1, t1, t2, 0)
             for t1, t2 in itertools.product(F, F)}
    if skip == "rotate":
        assert len(gains) > 1
    else:
        a = 1 * pow(1 - OMEGA, -1, P) % P
        assert gains == {extension(GATES["AND"], a, a)} != {1}


def _tree_cases():
    rng = random.Random(7)
    triples = [(f, g1, g2) for f in range(16) for g1, g2 in
               ((GATES["AND"], GATES["OR"]), (GATES["XOR"], GATES["NAND"]),
                (rng.randrange(16), rng.randrange(16)))]
    for f, g1, g2 in triples:
        for a, b, c, d in itertools.product(BITS, repeat=4):
            regs = {k: rng.randrange(P) for k in ("t1", "t2", "tl", "tr", "y")}
            yield f, g1, g2, a, b, c, d, regs


def test_gate_tree_clean():
    """f(g1(a, b), g2(c, d)): the output gains the tree's value, and all
    four mask registers come back (48 gate triples x 16 inputs)."""
    n = 0
    for f, g1, g2, a, b, c, d, regs in _tree_cases():
        out = gate_tree(f, g1, g2, a, b, c, d, regs)
        want = gate(f, gate(g1, a, b), gate(g2, c, d))
        assert out == {**regs, "y": (regs["y"] + want) % P}
        n += 1
    assert n == 48 * 16

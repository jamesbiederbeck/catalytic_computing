"""Tests for field.py — the foundation everything depends on."""

import pytest
from cmtz.field import (
    IRField, primitive_roots_Fp, generator_Fp, to_base,
    compute_working_field, horner_eval_Fq,
)


class TestIRField:
    def test_valid_construction(self):
        f = IRField(p=7, q=7)
        assert f.m == 6

    def test_p_not_prime_raises(self):
        with pytest.raises(ValueError, match="not prime"):
            IRField(p=6, q=7)

    def test_q_less_than_p_raises(self):
        with pytest.raises(ValueError, match="must be >= p"):
            IRField(p=7, q=5)

    def test_q_not_prime_raises(self):
        with pytest.raises(ValueError, match="not prime"):
            IRField(p=7, q=8)

    def test_frozen(self):
        f = IRField(p=7, q=7)
        with pytest.raises(AttributeError):
            f.p = 11  # type: ignore

    def test_validate_exponent(self):
        f = IRField(p=7, q=7)
        assert f.validate_exponent(8) == 2   # 8 mod 6 = 2
        assert f.validate_exponent(-1) == 5  # -1 mod 6 = 5


class TestPrimitiveRoots:
    def test_Fp7(self):
        """primitive_roots_Fp(7) should give a permutation of {1,2,3,4,5,6}."""
        roots = primitive_roots_Fp(7)
        assert len(roots) == 6
        assert set(roots) == {1, 2, 3, 4, 5, 6}
        # First element is ω^0 = 1
        assert roots[0] == 1

    def test_Fp5(self):
        roots = primitive_roots_Fp(5)
        assert len(roots) == 4
        assert set(roots) == {1, 2, 3, 4}
        assert roots[0] == 1

    def test_Fp17(self):
        roots = primitive_roots_Fp(17)
        assert len(roots) == 16
        assert set(roots) == set(range(1, 17))
        assert roots[0] == 1

    def test_generator_property(self):
        """Verify that successive powers of the generator produce all roots."""
        p = 13
        g = generator_Fp(p)
        roots = primitive_roots_Fp(p)
        for j in range(p - 1):
            assert roots[j] == pow(g, j, p)

    def test_not_prime_raises(self):
        with pytest.raises(ValueError):
            primitive_roots_Fp(10)

    def test_caching(self):
        """Verify lru_cache works (same object returned)."""
        r1 = primitive_roots_Fp(7)
        r2 = primitive_roots_Fp(7)
        assert r1 is r2


class TestToBase:
    def test_base2(self):
        assert to_base(13, 2) == [1, 0, 1, 1]  # 13 = 1 + 0*2 + 1*4 + 1*8

    def test_base10(self):
        assert to_base(123, 10) == [3, 2, 1]

    def test_zero(self):
        assert to_base(0, 2) == [0]

    def test_base_too_small_raises(self):
        with pytest.raises(ValueError):
            to_base(10, 1)


class TestComputeWorkingField:
    def test_small_case(self):
        q = compute_working_field(p=7, degree=2, n=4)
        # lower_bound = 2^4 * 7^3 = 16 * 343 = 5488
        assert q >= 5488
        import sympy
        assert sympy.isprime(q)

    def test_larger_case(self):
        q = compute_working_field(p=17, degree=8, n=16)
        # This should be astronomically large
        assert q >= 2**16 * 17**9


class TestHornerEval:
    def test_constant_poly(self):
        assert horner_eval_Fq([5], 3, 17) == 5

    def test_linear(self):
        # 3 + 2x at x=4 mod 17 = 3 + 8 = 11
        assert horner_eval_Fq([3, 2], 4, 17) == 11

    def test_quadratic(self):
        # 1 + 0x + 1x^2 at x=3 mod 17 = 1 + 9 = 10
        assert horner_eval_Fq([1, 0, 1], 3, 17) == 10

    def test_mod_reduction(self):
        # 10 + 10x at x=10 mod 17 = 10 + 100 = 110 mod 17 = 8
        assert horner_eval_Fq([10, 10], 10, 17) == 110 % 17

"""Test the critical v1→v2 cyclo fix.

v1 bug: cyclo(7) called sympy.cyclotomic_poly(7, x) which gives
  Φ_7(x) = x^6 - x^5 + x^4 - x^3 + x^2 - x + 1
This is the minimal polynomial over Q, NOT the list of primitive
roots in any specific field.

v2 fix: roots(7) computes {ω^j mod 7 : j ∈ [6]} = {1,3,2,6,4,5}
(for generator g=3). These are the actual field elements needed
for the ω-gate.
"""

import pytest
import sympy

from cmtz.field import primitive_roots_Fp, IRField
from cmtz.ir_nodes import IRPrimitiveRoots


class TestCycloFix:
    def test_v1_was_wrong(self):
        """Demonstrate what v1 computed (cyclotomic polynomial)
        vs what was actually needed (primitive roots)."""
        x = sympy.Symbol('x')
        # v1 computed this:
        phi7 = sympy.cyclotomic_poly(7, x)
        v1_coeffs = [int(c) for c in sympy.Poly(phi7, x).all_coeffs()]
        # Φ_7(x) = x^6 + x^5 + x^4 + x^3 + x^2 + x + 1
        # (7 is prime, so Φ_7 = (x^7 - 1)/(x - 1), all coefficients +1)
        assert v1_coeffs == [1, 1, 1, 1, 1, 1, 1]

        # v2 computes this:
        v2_roots = primitive_roots_Fp(7)
        # Should be a permutation of {1,2,3,4,5,6}
        assert set(v2_roots) == {1, 2, 3, 4, 5, 6}

        # These are completely different objects!
        # The polynomial coefficients tell you nothing about which
        # field to work in or what the actual root values are.

    def test_roots_are_powers_of_generator(self):
        """The root table must be successive powers of a generator."""
        p = 7
        roots = primitive_roots_Fp(p)
        g = roots[1]  # ω^1 = g (the generator)

        for j in range(p - 1):
            assert roots[j] == pow(g, j, p), (
                f"roots[{j}] = {roots[j]} but g^{j} mod {p} = {pow(g, j, p)}"
            )

    def test_roots_form_cyclic_group(self):
        """The roots must form the cyclic group F_p*."""
        for p in [5, 7, 11, 13, 17, 23]:
            roots = primitive_roots_Fp(p)
            assert len(roots) == p - 1
            assert set(roots) == set(range(1, p))
            # ω^0 = 1
            assert roots[0] == 1

    def test_IRPrimitiveRoots_build(self):
        """IRPrimitiveRoots.build() should produce the correct root table."""
        field = IRField(p=7, q=7)
        node = IRPrimitiveRoots.build(ir_field=field, name="test_roots")

        assert node.order == 6
        assert len(node.roots) == 6
        assert set(node.roots) == {1, 2, 3, 4, 5, 6}
        assert node.basic_instructions == 6
        assert node.ir_field == field

    def test_root_of_unity_cancellation(self):
        """Key property: sum of all m-th roots of unity = 0 mod p.
        This is what makes the Cook-Mertz polynomial evaluation work."""
        for p in [5, 7, 11, 13, 17]:
            roots = primitive_roots_Fp(p)
            total = sum(roots) % p
            assert total == 0, (
                f"Sum of roots mod {p} = {total}, expected 0"
            )

    def test_rotation_product_identity(self):
        """ω^m = ω^0 = 1 in F_p (the rotation wraps around)."""
        for p in [5, 7, 11, 13]:
            roots = primitive_roots_Fp(p)
            g = roots[1]  # generator
            m = p - 1
            assert pow(g, m, p) == 1

"""Tests for optimization passes: rotate fusion, CSE, constant folding."""

import pytest
from cmtz.field import IRField
from cmtz.ir_nodes import (
    IRProgram, IREmbed, IRRotate, IRPoly,
)
from cmtz.optimization.rotate_fusion import fuse_rotations
from cmtz.optimization.cse import eliminate_common_subexpressions
from cmtz.optimization.constant_fold import fold_constants


class TestRotateFusion:
    def test_fuse_adjacent(self):
        """Two adjacent rotates should fuse: ω^2 · ω^3 = ω^5."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)

        emb = IREmbed(name="emb", ir_field=field, index=0, psi=0)
        rot1 = IRRotate(name="r1", ir_field=field, j=2, src=emb, parents=[emb])
        rot2 = IRRotate(name="r2", ir_field=field, j=3, src=rot1, parents=[rot1])

        prog.add_node(emb)
        prog.add_node(rot1)
        prog.add_node(rot2)

        count = fuse_rotations(prog)
        assert count == 1
        # rot2 should now have j = (2+3) % 6 = 5
        remaining = prog.get_node("r2")
        assert remaining.j == 5

    def test_no_fuse_fanout(self):
        """Don't fuse if the first rotate has multiple children."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)

        emb = IREmbed(name="emb", ir_field=field, index=0, psi=0)
        rot1 = IRRotate(name="r1", ir_field=field, j=2, src=emb, parents=[emb])
        rot2 = IRRotate(name="r2", ir_field=field, j=3, src=rot1, parents=[rot1])
        rot3 = IRRotate(name="r3", ir_field=field, j=1, src=rot1, parents=[rot1])

        prog.add_node(emb)
        prog.add_node(rot1)
        prog.add_node(rot2)
        prog.add_node(rot3)

        count = fuse_rotations(prog)
        assert count == 0  # fan-out prevents fusion

    def test_fuse_wraps_mod_m(self):
        """Fusion wraps around: ω^4 · ω^5 = ω^(9 mod 6) = ω^3."""
        field = IRField(p=7, q=7)  # m = 6
        prog = IRProgram(ir_field=field)

        emb = IREmbed(name="emb", ir_field=field, index=0, psi=0)
        rot1 = IRRotate(name="r1", ir_field=field, j=4, src=emb, parents=[emb])
        rot2 = IRRotate(name="r2", ir_field=field, j=5, src=rot1, parents=[rot1])

        prog.add_node(emb)
        prog.add_node(rot1)
        prog.add_node(rot2)

        fuse_rotations(prog)
        assert prog.get_node("r2").j == (4 + 5) % 6  # = 3


class TestCSE:
    def test_deduplicate_identical_polys(self):
        """Two IRPoly nodes with same coeffs and field should merge."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)

        emb = IREmbed(name="emb", ir_field=field, index=0, psi=0)
        p1 = IRPoly(name="p1", ir_field=field, coeffs=[1, 2, 3], degree=2,
                     parents=[emb])
        p2 = IRPoly(name="p2", ir_field=field, coeffs=[1, 2, 3], degree=2,
                     parents=[emb])

        prog.add_node(emb)
        prog.add_node(p1)
        prog.add_node(p2)

        count = eliminate_common_subexpressions(prog)
        assert count == 1
        assert len(prog) == 2  # emb + one poly

    def test_different_coeffs_not_merged(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)

        emb = IREmbed(name="emb", ir_field=field, index=0, psi=0)
        p1 = IRPoly(name="p1", ir_field=field, coeffs=[1, 2, 3], degree=2,
                     parents=[emb])
        p2 = IRPoly(name="p2", ir_field=field, coeffs=[1, 2, 4], degree=2,
                     parents=[emb])

        prog.add_node(emb)
        prog.add_node(p1)
        prog.add_node(p2)

        count = eliminate_common_subexpressions(prog)
        assert count == 0

    def test_different_field_not_merged(self):
        f1 = IRField(p=7, q=7)
        f2 = IRField(p=7, q=11)
        prog = IRProgram(ir_field=f1)

        emb = IREmbed(name="emb", ir_field=f1, index=0, psi=0)
        p1 = IRPoly(name="p1", ir_field=f1, coeffs=[1, 2], degree=1,
                     parents=[emb])
        p2 = IRPoly(name="p2", ir_field=f2, coeffs=[1, 2], degree=1,
                     parents=[emb])

        prog.add_node(emb)
        prog.add_node(p1)
        prog.add_node(p2)

        count = eliminate_common_subexpressions(prog)
        assert count == 0


class TestConstantFold:
    def test_fold_constant_poly(self):
        """A poly whose only input is a constant embed should fold."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)

        emb = IREmbed(name="emb", ir_field=field, index=0, psi=2)
        # poly: 1 + 2x + 3x^2 at x = omega^2
        poly = IRPoly(name="p1", ir_field=field, coeffs=[1, 2, 3], degree=2,
                      parents=[emb])

        prog.add_node(emb)
        prog.add_node(poly)

        count = fold_constants(prog)
        assert count == 1
        # The poly node should now be an IREmbed
        folded = prog.get_node("p1")
        assert isinstance(folded, IREmbed)

    def test_no_fold_non_constant(self):
        """Poly with non-embed parent should not fold."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)

        emb = IREmbed(name="emb", ir_field=field, index=0, psi=0)
        rot = IRRotate(name="rot", ir_field=field, j=1, src=emb, parents=[emb])
        poly = IRPoly(name="p1", ir_field=field, coeffs=[1, 2], degree=1,
                      parents=[rot])

        prog.add_node(emb)
        prog.add_node(rot)
        prog.add_node(poly)

        count = fold_constants(prog)
        assert count == 0
        assert isinstance(prog.get_node("p1"), IRPoly)

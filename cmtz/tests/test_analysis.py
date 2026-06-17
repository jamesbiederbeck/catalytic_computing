"""Tests for analysis passes: cycle check, field consistency, catalytic verify."""

import pytest
from cmtz.field import IRField, find_nonresidue, add_Fp2, conj_Fp2, magsq_Fp2
from cmtz.ir_nodes import (
    IRProgram, IRNode, IREmbed, IRRotate, IRCompose,
    IRRegion, IRCatalyticRegion, CycleError,
    IRAdd, IRComplexEmbed, IRConjugate, IRMagnitudeSq,
)
from cmtz.analysis.cycle_check import (
    check_cycles, check_cycles_diagnostic, find_back_edges,
)
from cmtz.analysis.field_consistency import (
    check_field_consistency, FieldMismatchError,
)
from cmtz.analysis.catalytic_verify import (
    verify_catalytic, verify_region, CatalyticViolationError,
)
from cmtz.analysis.cost_propagation import (
    propagate_costs, check_matpow_budget,
)


class TestCycleCheck:
    def test_linear_dag(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        b = IRRotate(name="b", ir_field=field, j=1, src=a, parents=[a])
        prog.add_node(a)
        prog.add_node(b)
        order = check_cycles(prog)
        names = [n for n, _ in order]
        assert names.index("a") < names.index("b")

    def test_cycle_detected(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        b = IRRotate(name="b", ir_field=field, j=1, src=a, parents=[a])
        # Create cycle: a depends on b which depends on a
        a.parents = [b]
        prog.add_node(a)
        prog.add_node(b)
        with pytest.raises(CycleError):
            check_cycles(prog)

    def test_diagnostic_acyclic(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        prog.add_node(a)
        result = check_cycles_diagnostic(prog)
        assert result.is_dag
        assert result.node_count == 1
        assert result.depth == 0

    def test_diagnostic_cycle(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        b = IRRotate(name="b", ir_field=field, j=1, src=a, parents=[a])
        a.parents = [b]
        prog.add_node(a)
        prog.add_node(b)
        result = check_cycles_diagnostic(prog)
        assert not result.is_dag
        assert result.cycle_node is not None

    def test_find_back_edges_clean(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        b = IRRotate(name="b", ir_field=field, j=1, src=a, parents=[a])
        prog.add_node(a)
        prog.add_node(b)
        edges = find_back_edges(prog)
        assert edges == []

    def test_find_back_edges_cycle(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        b = IRRotate(name="b", ir_field=field, j=1, src=a, parents=[a])
        a.parents = [b]
        prog.add_node(a)
        prog.add_node(b)
        edges = find_back_edges(prog)
        assert len(edges) > 0

    def test_depth_calculation(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        b = IRRotate(name="b", ir_field=field, j=1, src=a, parents=[a])
        c = IRRotate(name="c", ir_field=field, j=2, src=b, parents=[b])
        prog.add_node(a)
        prog.add_node(b)
        prog.add_node(c)
        result = check_cycles_diagnostic(prog)
        assert result.depth == 2  # a(0) -> b(1) -> c(2)


class TestFieldConsistency:
    def test_consistent_field(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        b = IRRotate(name="b", ir_field=field, j=1, src=a, parents=[a])
        prog.add_node(a)
        prog.add_node(b)
        check_field_consistency(prog)  # should not raise

    def test_mismatched_field(self):
        f1 = IRField(p=7, q=7)
        f2 = IRField(p=11, q=11)
        prog = IRProgram(ir_field=f1)
        a = IREmbed(name="a", ir_field=f1, index=0, psi=0)
        b = IRRotate(name="b", ir_field=f2, j=1, src=a, parents=[a])
        prog.add_node(a)
        prog.add_node(b)
        with pytest.raises(FieldMismatchError):
            check_field_consistency(prog)

    def test_none_field_skipped(self):
        """Nodes with no field annotation should be skipped."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        a = IREmbed(name="a", ir_field=field, index=0, psi=0)
        b = IRNode(name="b", parents=[a])  # no field
        prog.add_node(a)
        prog.add_node(b)
        check_field_consistency(prog)  # should not raise


class TestCatalyticVerify:
    def test_empty_region(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        region = IRCatalyticRegion(
            name="cat1", ir_field=field,
            inner_nodes=[], catalytic_regs=["r0"],
        )
        prog.add_node(region)
        deltas = verify_catalytic(prog)
        assert len(deltas) == 1
        assert deltas[0].is_restored

    def test_region_with_clean_inner(self):
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        inner = IREmbed(name="inner_emb", ir_field=field, index=0, psi=0)
        prog.add_node(inner)
        region = IRCatalyticRegion(
            name="cat1", ir_field=field,
            inner_nodes=[inner], catalytic_regs=["r0", "r1"],
        )
        prog.add_node(region)
        deltas = verify_catalytic(prog)
        assert all(d.is_restored for d in deltas)

    def test_verify_single_region(self):
        field = IRField(p=7, q=7)
        region = IRCatalyticRegion(
            name="cat1", ir_field=field,
            inner_nodes=[], catalytic_regs=["r0"],
        )
        deltas = verify_region(region)
        assert len(deltas) == 1
        assert deltas[0].register == "r0"
        assert deltas[0].delta_expr == "0"

    def test_irregion_skipped_by_verifier(self):
        """IRRegion (advisory/catalytic {}) nodes are not verified."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        # IRRegion with an embed that would clobber r0 — fine, not checked
        inner = IREmbed(name="r0", ir_field=field, index=0, psi=5)
        prog.add_node(inner)
        region = IRRegion(
            name="adv1", ir_field=field,
            inner_nodes=[inner], catalytic_regs=["r0"],
        )
        prog.add_node(region)
        deltas = verify_catalytic(prog)
        assert deltas == []  # no strict regions → nothing verified

    def test_embed_clobber_raises(self):
        """IREmbed whose name matches a catalytic reg is a violation in IRCatalyticRegion."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        # embed_0 is both the inner node output name and the listed catalytic reg
        inner = IREmbed(name="embed_0", ir_field=field, index=0, psi=3)
        prog.add_node(inner)
        region = IRCatalyticRegion(
            name="handler1", ir_field=field,
            inner_nodes=[inner], catalytic_regs=["embed_0"],
        )
        prog.add_node(region)
        with pytest.raises(CatalyticViolationError):
            verify_catalytic(prog)


class TestCostPropagation:
    def test_compose_lemma_2_1(self):
        """Verify Lemma 2.1: t(f∘g) = t_f * t_g, s(f∘g) = s_f + t_f*s_g."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)

        f = IREmbed(name="f", ir_field=field, index=0, psi=0)
        f.recursive_calls = 3
        f.basic_instructions = 10
        f.num_registers = 4

        g = IREmbed(name="g", ir_field=field, index=1, psi=0)
        g.recursive_calls = 2
        g.basic_instructions = 5
        g.num_registers = 6

        comp = IRCompose(name="comp", ir_field=field, parents=[f, g])
        prog.add_node(f)
        prog.add_node(g)
        prog.add_node(comp)

        propagate_costs(prog)

        assert comp.recursive_calls == 3 * 2  # t_f * t_g
        assert comp.basic_instructions == 10 + 3 * 5  # s_f + t_f * s_g
        assert comp.num_registers == 6  # max(4, 6)


class TestComplexField:
    """Tests for F_{p²} field extension and arithmetic helpers."""

    def test_find_nonresidue_p7(self):
        c = find_nonresidue(7)
        # c^((7-1)/2) ≡ -1 (mod 7) i.e. ≡ 6
        assert pow(c, 3, 7) == 6

    def test_find_nonresidue_p17(self):
        c = find_nonresidue(17)
        assert pow(c, 8, 17) == 16

    def test_irfield_complex_valid(self):
        c = find_nonresidue(7)
        f = IRField(p=7, q=53, c=c)
        assert f.is_complex
        assert f.c == c

    def test_irfield_complex_invalid_c(self):
        """c=1 is not a non-residue mod 7 (1^3 = 1, not -1)."""
        with pytest.raises(ValueError, match="non-residue"):
            IRField(p=7, q=53, c=1)

    def test_irfield_real_field(self):
        c = find_nonresidue(7)
        cfield = IRField(p=7, q=53, c=c)
        real = cfield.real_field()
        assert not real.is_complex
        assert real.p == 7
        assert real.q == 53

    def test_add_Fp2(self):
        assert add_Fp2((2, 3), (4, 5), 7) == (6, 1)  # (6, 8 mod 7)

    def test_conj_Fp2(self):
        assert conj_Fp2((3, 2), 7) == (3, 5)  # 3 + 2i → 3 - 2i = 3 + 5i mod 7

    def test_magsq_Fp2_identity(self):
        """For (1, 0), |1+0i|² = 1."""
        c = find_nonresidue(7)
        assert magsq_Fp2((1, 0), 7, c) == 1

    def test_magsq_Fp2_formula(self):
        """Verify |a+bi|² = a² - b²·c mod p."""
        p, c = 7, find_nonresidue(7)
        a, b = 3, 2
        expected = (a*a - b*b*c) % p
        assert magsq_Fp2((a, b), p, c) == expected


class TestComplexFieldConsistency:
    """Tests for v4 field consistency rules (§6.5)."""

    def test_magsq_requires_complex_input(self):
        from cmtz.analysis.field_consistency import check_field_consistency, FieldMismatchError
        real_field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=real_field)
        embed = IREmbed(name="e", ir_field=real_field, index=0, psi=0)
        msq = IRMagnitudeSq(name="msq", ir_field=real_field, parents=[embed])
        prog.add_node(embed)
        prog.add_node(msq)
        with pytest.raises(FieldMismatchError, match="complex"):
            check_field_consistency(prog)

    def test_magsq_output_must_be_real_field(self):
        from cmtz.analysis.field_consistency import check_field_consistency, FieldMismatchError
        c = find_nonresidue(7)
        cfield = IRField(p=7, q=53, c=c)
        prog = IRProgram(ir_field=cfield)
        cembed = IRComplexEmbed(name="ce", ir_field=cfield, re_psi=0, im_psi=1)
        # Wrong: output ir_field is still complex instead of real
        msq = IRMagnitudeSq(name="msq", ir_field=cfield, parents=[cembed])
        prog.add_node(cembed)
        prog.add_node(msq)
        with pytest.raises(FieldMismatchError):
            check_field_consistency(prog)

    def test_magsq_correct_type_lowering(self):
        from cmtz.analysis.field_consistency import check_field_consistency
        c = find_nonresidue(7)
        cfield = IRField(p=7, q=53, c=c)
        real_field = cfield.real_field()
        prog = IRProgram(ir_field=cfield)
        cembed = IRComplexEmbed(name="ce", ir_field=cfield, re_psi=0, im_psi=1)
        msq = IRMagnitudeSq(name="msq", ir_field=real_field, parents=[cembed])
        prog.add_node(cembed)
        prog.add_node(msq)
        check_field_consistency(prog)  # should not raise

    def test_conj_requires_complex_field(self):
        from cmtz.analysis.field_consistency import check_field_consistency, FieldMismatchError
        real_field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=real_field)
        embed = IREmbed(name="e", ir_field=real_field, index=0, psi=0)
        conj = IRConjugate(name="c", ir_field=real_field, parents=[embed])
        prog.add_node(embed)
        prog.add_node(conj)
        with pytest.raises(FieldMismatchError, match="complex"):
            check_field_consistency(prog)

    def test_add_same_field_passes(self):
        from cmtz.analysis.field_consistency import check_field_consistency
        c = find_nonresidue(7)
        cfield = IRField(p=7, q=53, c=c)
        prog = IRProgram(ir_field=cfield)
        a = IRComplexEmbed(name="a", ir_field=cfield, re_psi=0, im_psi=1)
        b = IRComplexEmbed(name="b", ir_field=cfield, re_psi=2, im_psi=0)
        add = IRAdd(name="ab", ir_field=cfield, parents=[a, b])
        prog.add_node(a)
        prog.add_node(b)
        prog.add_node(add)
        check_field_consistency(prog)  # should not raise

    def test_add_mixed_fields_raises(self):
        from cmtz.analysis.field_consistency import check_field_consistency, FieldMismatchError
        c = find_nonresidue(7)
        cfield = IRField(p=7, q=53, c=c)
        real_field = cfield.real_field()
        prog = IRProgram(ir_field=cfield)
        a = IRComplexEmbed(name="a", ir_field=cfield, re_psi=0, im_psi=1)
        b = IREmbed(name="b", ir_field=real_field, index=0, psi=0)
        add = IRAdd(name="ab", ir_field=cfield, parents=[a, b])
        prog.add_node(a)
        prog.add_node(b)
        prog.add_node(add)
        with pytest.raises(FieldMismatchError):
            check_field_consistency(prog)

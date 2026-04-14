"""Tests for analysis passes: cycle check, field consistency, catalytic verify."""

import pytest
from cmtz.field import IRField
from cmtz.ir_nodes import (
    IRProgram, IRNode, IREmbed, IRRotate, IRCompose,
    IRCatalyticRegion, CycleError,
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

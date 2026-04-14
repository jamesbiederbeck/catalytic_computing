"""AST → IR lowering for Cook-Mertz v2.

Transforms the elaborated AST into an IR DAG with:
  - Field annotations on every node
  - cyclo(p) → IRPrimitiveRoots (not cyclotomic polynomial)
  - Catalytic regions wrapped in IRCatalyticRegion
  - Composition via IRCompose with Lemma 2.1 cost accounting
"""

from __future__ import annotations

from .ast_nodes import (
    Program, EmbedStmt, RotateStmt, ComposeStmt, MeasureStmt,
    RootsStmt, CycloPhiStmt, MatPowStmt, CatalyticStmt, ASTNode,
)
from .ir_nodes import (
    IRProgram, IRNode, IREmbed, IRPrimitiveRoots, IRRotate,
    IRPoly, IRMatPow, IRCatalyticRegion, IRCompose, IRMeasure,
    IRCycloPhiPoly,
)
from .field import IRField

import sympy


class LoweringError(Exception):
    pass


class Lowering:
    """Lower AST to IR DAG."""

    def __init__(self, ir_field: IRField | None):
        self.ir_field = ir_field
        self.program = IRProgram(ir_field=ir_field)
        self._name_counter = 0

    def _fresh_name(self, prefix: str) -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def lower(self, ast: Program) -> IRProgram:
        """Lower a complete program AST to IR."""
        for stmt in ast.statements:
            self._lower_stmt(stmt)
        return self.program

    def _lower_stmt(self, stmt: ASTNode) -> IRNode | None:
        if isinstance(stmt, EmbedStmt):
            return self._lower_embed(stmt)
        elif isinstance(stmt, RotateStmt):
            return self._lower_rotate(stmt)
        elif isinstance(stmt, ComposeStmt):
            return self._lower_compose(stmt)
        elif isinstance(stmt, MeasureStmt):
            return self._lower_measure(stmt)
        elif isinstance(stmt, RootsStmt):
            return self._lower_roots(stmt)
        elif isinstance(stmt, CycloPhiStmt):
            return self._lower_cyclo_phi(stmt)
        elif isinstance(stmt, MatPowStmt):
            return self._lower_matpow(stmt)
        elif isinstance(stmt, CatalyticStmt):
            return self._lower_catalytic(stmt)
        else:
            raise LoweringError(f"Unknown AST node type: {type(stmt).__name__}")

    def _lower_embed(self, stmt: EmbedStmt) -> IREmbed:
        name = f"embed_{stmt.index}"
        node = IREmbed(
            name=name,
            ir_field=self.ir_field,
            index=stmt.index,
            psi=stmt.psi,
        )
        self.program.add_node(node)
        return node

    def _lower_rotate(self, stmt: RotateStmt) -> IRRotate:
        src_node = self.program.get_node(stmt.src)
        name = f"rot_{stmt.src}_{stmt.dst}"
        j = stmt.j
        if self.ir_field is not None:
            j = self.ir_field.validate_exponent(j)
        node = IRRotate(
            name=name,
            ir_field=self.ir_field,
            j=j,
            src=src_node,
            parents=[src_node],
        )
        self.program.add_node(node)
        return node

    def _lower_compose(self, stmt: ComposeStmt) -> IRCompose:
        """Compose pipeline stages with Lemma 2.1 cost accounting."""
        if len(stmt.pipeline) < 2:
            raise LoweringError("compose requires at least 2 stages")

        nodes = [self.program.get_node(name) for name in stmt.pipeline]

        # Pairwise composition with Lemma 2.1
        result = nodes[0]
        for i in range(1, len(nodes)):
            g = nodes[i]
            name = self._fresh_name("compose")
            comp = IRCompose(
                name=name,
                ir_field=self.ir_field,
                parents=[result, g],
            )
            # Lemma 2.1 cost propagation
            comp.recursive_calls = max(1, result.recursive_calls) * max(1, g.recursive_calls)
            comp.basic_instructions = (
                result.basic_instructions +
                max(1, result.recursive_calls) * g.basic_instructions
            )
            comp.num_registers = max(result.num_registers, g.num_registers)
            self.program.add_node(comp)
            result = comp

        return result

    def _lower_roots(self, stmt: RootsStmt) -> IRPrimitiveRoots:
        """Lower roots(p) to IRPrimitiveRoots.

        This is the critical v1 fix: instead of computing the cyclotomic
        polynomial Φ_p(x), we enumerate the actual primitive roots
        {ω^0, ω^1, ..., ω^(m-1)} in F_p.
        """
        field = self.ir_field
        if field is None:
            field = IRField(p=stmt.p, q=stmt.p)
            self.ir_field = field

        node = IRPrimitiveRoots.build(ir_field=field, name=stmt.name)
        self.program.add_node(node)
        return node

    def _lower_cyclo_phi(self, stmt: CycloPhiStmt) -> IRCycloPhiPoly:
        """Lower cyclo_phi(n) — the actual cyclotomic polynomial.
        Kept for FIR filter use case, but renamed from v1's cyclo()."""
        x = sympy.Symbol('x')
        poly = sympy.cyclotomic_poly(stmt.n, x)
        coeffs = [int(c) for c in sympy.Poly(poly, x).all_coeffs()[::-1]]

        node = IRCycloPhiPoly(
            name=stmt.name,
            ir_field=self.ir_field,
            n=stmt.n,
            poly_coeffs=coeffs,
        )
        node.basic_instructions = len(coeffs)
        self.program.add_node(node)
        return node

    def _lower_matpow(self, stmt: MatPowStmt) -> IRMatPow:
        if self.ir_field is None:
            raise LoweringError("matpow requires a field declaration")

        matrix_node = self.program.get_node(stmt.matrix)
        node = IRMatPow(
            name=stmt.name,
            ir_field=self.ir_field,
            matrix_name=stmt.matrix,
            d=stmt.d,
            epsilon=stmt.epsilon,
            n_dim=0,  # resolved at backend time from matrix shape
            parents=[matrix_node],
        )
        self.program.add_node(node)
        return node

    def _lower_catalytic(self, stmt: CatalyticStmt) -> IRCatalyticRegion:
        """Lower catalytic { body } restoring (regs).

        Wraps the body in an IRCatalyticRegion and marks the
        restoring registers as catalytic.
        """
        inner_nodes = []
        for inner_stmt in stmt.body:
            ir_node = self._lower_stmt(inner_stmt)
            if ir_node is not None:
                inner_nodes.append(ir_node)

        # Mark restoring registers as catalytic
        for reg_name in stmt.restoring:
            try:
                reg_node = self.program.get_node(reg_name)
                reg_node.is_catalytic = True
            except KeyError:
                pass  # register may be defined outside this scope

        name = self._fresh_name("catalytic")
        region = IRCatalyticRegion(
            name=name,
            ir_field=self.ir_field,
            inner_nodes=inner_nodes,
            catalytic_regs=stmt.restoring,
        )
        self.program.add_node(region)
        return region

    def _lower_measure(self, stmt: MeasureStmt) -> IRMeasure:
        src_node = self.program.get_node(stmt.src)
        node = IRMeasure(
            name=stmt.name,
            ir_field=self.ir_field,
            mode=stmt.mode,
            src=src_node,
            parents=[src_node],
        )
        self.program.add_node(node)
        return node


def lower(ast: Program, ir_field: IRField | None = None) -> IRProgram:
    """Convenience function: lower an AST to an IR program."""
    l = Lowering(ir_field)
    return l.lower(ast)

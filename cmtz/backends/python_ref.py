"""Python reference backend for Cook-Mertz v4.

Exact modular arithmetic over Z_p — no floating point anywhere.
This is the ground-truth backend for correctness testing.

All operations:
  - IRRotate: multiply by ω^j mod p
  - IRPoly: Horner evaluation mod q, then reduce mod p
  - IRMeasure mod_p: return value in F_p

v4 additions (complex F_{p²} support):
  - Registers are int (real) or (int, int) tuple (complex)
  - IRAdd: add_Fp2 for complex, plain addition for real
  - IRComplexEmbed: initializes (ω^re_psi, ω^im_psi)
  - IRConjugate: conj_Fp2
  - IRMagnitudeSq: magsq_Fp2 → int
"""

from __future__ import annotations

from ..ir_nodes import (
    IRProgram, IRNode, IREmbed, IRPrimitiveRoots, IRRotate,
    IRPoly, IRMatPow, IRMeasure, IRCompose, IRRegion, IRCatalyticRegion,
    IRAdd, IRComplexEmbed, IRConjugate, IRMagnitudeSq,
)
from ..field import (
    IRField, primitive_roots_Fp, horner_eval_Fq, to_base,
    add_Fp2, conj_Fp2, magsq_Fp2,
)


class PythonBackend:
    """Reference evaluator using exact Python integer arithmetic."""

    def __init__(self, ir_field: IRField):
        self.field = ir_field
        self.p = ir_field.p
        self.q = ir_field.q
        self.roots = primitive_roots_Fp(ir_field.p)
        self.registers: dict[str, int] = {}

    def eval_program(self, prog: IRProgram) -> dict[str, int]:
        """Evaluate an IR program, returning all named results."""
        for name, node in prog.topo_order():
            self.registers[name] = self._eval_node(node)
        return dict(self.registers)

    def _eval_node(self, node: IRNode) -> int | tuple[int, int]:
        if isinstance(node, IREmbed):
            return self.eval_embed(node)
        elif isinstance(node, IRPrimitiveRoots):
            return 0  # root table, not a value
        elif isinstance(node, IRRotate):
            return self.eval_rotate(node)
        elif isinstance(node, IRPoly):
            return self.eval_poly(node)
        elif isinstance(node, IRMeasure):
            return self.eval_measure(node)
        elif isinstance(node, IRCompose):
            return self.eval_compose(node)
        elif isinstance(node, IRRegion):
            return self.eval_catalytic(node)
        elif isinstance(node, IRCatalyticRegion):
            return self.eval_catalytic(node)
        elif isinstance(node, IRMatPow):
            return 0  # matpow needs matrix context, handled separately
        elif isinstance(node, IRAdd):
            return self.eval_add(node)
        elif isinstance(node, IRComplexEmbed):
            return self.eval_complex_embed(node)
        elif isinstance(node, IRConjugate):
            return self.eval_conjugate(node)
        elif isinstance(node, IRMagnitudeSq):
            return self.eval_magsq(node)
        else:
            return 0

    def eval_embed(self, node: IREmbed) -> int:
        """Initialize register with ω^psi mod p."""
        if node.psi == 0:
            return 1  # ω^0 = 1
        return self.roots[node.psi % len(self.roots)]

    def eval_rotate(self, node: IRRotate) -> int:
        """Multiply register value by ω^j in F_p."""
        reg_val = self._get_parent_val(node)
        omega_j = self.roots[node.j % len(self.roots)]
        return (reg_val * omega_j) % self.p

    def eval_poly(self, node: IRPoly) -> int:
        """Horner evaluation in F_q, then reduce mod p."""
        x = self._get_parent_val(node)
        result = horner_eval_Fq(node.coeffs, x, self.q)
        return result % self.p

    def eval_measure(self, node: IRMeasure) -> int:
        """Extract scalar. For mod_p, return value reduced to F_p."""
        val = self._get_parent_val(node)
        if node.mode == 'mod_p':
            return val % self.p
        elif node.mode == 'real':
            return val  # in exact arithmetic, value IS the real part
        elif node.mode == 'mag':
            return abs(val) % self.p
        return val % self.p

    def eval_compose(self, node: IRCompose) -> int:
        """Composition result is the last parent's value."""
        if node.parents:
            return self._get_parent_val_by_index(node, -1)
        return 0

    def eval_catalytic(self, node: IRRegion | IRCatalyticRegion) -> int:
        """Evaluate inner nodes; return last inner node's value."""
        result = 0
        for inner in node.inner_nodes:
            result = self._eval_node(inner)
            self.registers[inner.name] = result
        return result

    def eval_add(self, node: IRAdd) -> int | tuple[int, int]:
        """Addition: (a + b) mod p for real, add_Fp2 for complex."""
        a = self._get_parent_val(node)
        b = self._get_parent_val_by_index(node, 1)
        if isinstance(a, tuple) and isinstance(b, tuple):
            return add_Fp2(a, b, self.p)
        return (a + b) % self.p  # type: ignore[operator]

    def eval_complex_embed(self, node: IRComplexEmbed) -> tuple[int, int]:
        """Initialize complex register ω^re_psi + i·ω^im_psi."""
        roots = self.roots
        m = len(roots)
        re = roots[node.re_psi % m]
        im = roots[node.im_psi % m]
        return (re, im)

    def eval_conjugate(self, node: IRConjugate) -> tuple[int, int]:
        """Complex conjugate: a+bi → a-bi mod p."""
        val = self._get_parent_val(node)
        if not isinstance(val, tuple):
            raise TypeError(
                f"IRConjugate '{node.name}': parent is not a complex value"
            )
        return conj_Fp2(val, self.p)

    def eval_magsq(self, node: IRMagnitudeSq) -> int:
        """Magnitude squared: |a+bi|² = a² - b²·c mod p."""
        val = self._get_parent_val(node)
        if not isinstance(val, tuple):
            raise TypeError(
                f"IRMagnitudeSq '{node.name}': parent is not a complex value"
            )
        parent = node.parents[0]
        if parent.ir_field is None or parent.ir_field.c is None:
            raise ValueError(
                f"IRMagnitudeSq '{node.name}': parent has no non-residue c"
            )
        return magsq_Fp2(val, self.p, parent.ir_field.c)

    def eval_matpow_matrix(self, node: IRMatPow,
                           matrix: list[list[int]]) -> list[list[int]]:
        """Compute M^d over F_p using base-δ decomposition (Lemma 3.12).

        matrix: n×n list-of-lists with integer entries in [0, p).
        Returns M^d mod p.
        """
        n = len(matrix)
        delta = node.delta
        digits = to_base(node.d, delta)

        # Start with identity
        result = [[int(i == j) for j in range(n)] for i in range(n)]
        Mk = [row[:] for row in matrix]  # copy

        for alpha in digits:
            if alpha > 0:
                # Compute Mk^alpha via repeated squaring in F_p
                block = self._mat_pow_small(Mk, alpha, n)
                result = self._mat_mul(result, block, n)
            # Advance: Mk ← Mk^delta
            Mk = self._mat_pow_small(Mk, delta, n)

        return result

    def _mat_mul(self, A: list[list[int]], B: list[list[int]],
                 n: int) -> list[list[int]]:
        """Matrix multiply mod p."""
        C = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                for k in range(n):
                    s += A[i][k] * B[k][j]
                C[i][j] = s % self.p
        return C

    def _mat_pow_small(self, M: list[list[int]], k: int,
                       n: int) -> list[list[int]]:
        """Compute M^k mod p by repeated squaring."""
        if k == 0:
            return [[int(i == j) for j in range(n)] for i in range(n)]
        if k == 1:
            return [row[:] for row in M]

        result = [[int(i == j) for j in range(n)] for i in range(n)]
        base = [row[:] for row in M]

        while k > 0:
            if k & 1:
                result = self._mat_mul(result, base, n)
            base = self._mat_mul(base, base, n)
            k >>= 1

        return result

    def _get_parent_val(self, node: IRNode) -> int:
        """Get the value of a node's first parent."""
        if node.parents:
            return self.registers.get(node.parents[0].name, 0)
        return 0

    def _get_parent_val_by_index(self, node: IRNode, idx: int) -> int:
        """Get the value of a parent by index."""
        if node.parents:
            return self.registers.get(node.parents[idx].name, 0)
        return 0

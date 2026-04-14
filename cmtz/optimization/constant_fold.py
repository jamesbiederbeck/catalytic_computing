"""Constant folding for Cook-Mertz v2 IR.

When all inputs to an IRPoly or IRRotate are compile-time constants,
evaluate the node and replace it with an IREmbed carrying the result.
"""

from __future__ import annotations

from ..ir_nodes import IRProgram, IRNode, IRPoly, IRRotate, IREmbed
from ..field import horner_eval_Fq, primitive_roots_Fp


def fold_constants(prog: IRProgram) -> int:
    """Evaluate constant-input nodes at compile time.

    Returns the number of nodes folded.
    """
    folded = 0

    for name, node in list(prog.nodes()):
        if isinstance(node, IRPoly) and node.ir_field is not None:
            # If this poly has no variable inputs (all parents are
            # constant embeds), we can evaluate it now.
            if _all_parents_constant(node):
                val = _eval_poly_constant(node)
                replacement = IREmbed(
                    name=name,
                    ir_field=node.ir_field,
                    index=0,
                    psi=val,
                )
                replacement.parents = []
                prog._nodes[name] = replacement
                folded += 1

    return folded


def _all_parents_constant(node: IRNode) -> bool:
    """Check if all parent nodes are constant (IREmbed with known psi)."""
    return all(isinstance(p, IREmbed) for p in node.parents)


def _eval_poly_constant(node: IRPoly) -> int:
    """Evaluate a constant polynomial in the working field."""
    if not node.parents or node.ir_field is None:
        return 0
    # Use the first parent's psi as the evaluation point
    parent = node.parents[0]
    if isinstance(parent, IREmbed):
        x = parent.psi % node.ir_field.q
        result = horner_eval_Fq(node.coeffs, x, node.ir_field.q)
        return result % node.ir_field.p
    return 0

"""Common subexpression elimination for IRPoly nodes.

Two IRPoly nodes with identical coefficients and field can share
a single computation. This is particularly useful after matpow
decomposition, where the same polynomial may appear multiple times.
"""

from __future__ import annotations

from ..ir_nodes import IRProgram, IRPoly


def eliminate_common_subexpressions(prog: IRProgram) -> int:
    """Deduplicate identical IRPoly nodes.

    Returns the number of nodes eliminated.
    """
    # Build signature → canonical node map
    seen: dict[tuple, IRPoly] = {}
    redirect: dict[str, str] = {}
    eliminated = 0

    for name, node in prog.nodes():
        if not isinstance(node, IRPoly):
            continue
        if node.ir_field is None:
            continue

        sig = (tuple(node.coeffs), node.ir_field.p, node.ir_field.q)
        if sig in seen:
            canonical = seen[sig]
            redirect[name] = canonical.name
            eliminated += 1
        else:
            seen[sig] = node

    # Redirect parent references
    if redirect:
        for name, node in prog.nodes():
            node.parents = [
                prog.get_node(redirect.get(p.name, p.name))
                for p in node.parents
            ]

        # Remove eliminated nodes
        for name in redirect:
            if name in prog._nodes:
                del prog._nodes[name]

    return eliminated

"""Adjacent rotation fusion for Cook-Mertz v2.

Two consecutive rotations by ω^j1 and ω^j2 on the same register
are equivalent to a single rotation by ω^(j1 + j2 mod m).
This reduces the number of modular multiplications.
"""

from __future__ import annotations

from ..ir_nodes import IRProgram, IRNode, IRRotate


def fuse_rotations(prog: IRProgram) -> int:
    """Fuse adjacent IRRotate nodes in the IR DAG.

    Returns the number of fusions performed.
    This is a conservative pass: only fuses when the second rotate's
    sole parent is the first rotate (no fan-out).
    """
    fused_count = 0
    nodes_to_remove: list[str] = []

    topo = prog.topo_order()

    # Build reverse map: node → children that reference it
    children_of: dict[str, list[tuple[str, IRNode]]] = {}
    for name, node in topo:
        for p in node.parents:
            children_of.setdefault(p.name, []).append((name, node))

    for name, node in topo:
        if not isinstance(node, IRRotate):
            continue
        if node.src is None or not isinstance(node.src, IRRotate):
            continue

        parent_rot = node.src
        # Only fuse if parent has exactly one child (no fan-out)
        parent_children = children_of.get(parent_rot.name, [])
        if len(parent_children) != 1:
            continue

        # Same field required
        if node.ir_field != parent_rot.ir_field:
            continue

        # Fuse: new j = (j_parent + j_child) mod m
        m = node.ir_field.m
        node.j = (parent_rot.j + node.j) % m
        node.src = parent_rot.src
        node.parents = [p for p in node.parents if p is not parent_rot]
        if parent_rot.src is not None:
            node.parents.insert(0, parent_rot.src)

        nodes_to_remove.append(parent_rot.name)
        fused_count += 1

    # Remove fused-away nodes
    for name in nodes_to_remove:
        if name in prog._nodes:
            del prog._nodes[name]

    return fused_count

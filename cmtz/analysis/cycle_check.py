"""Cycle detection and topological ordering for Cook-Mertz v2 IR.

The IR must be a DAG — cycles indicate invalid recursion that the
branch-free model cannot express. This pass runs first because every
subsequent analysis (field consistency, catalytic verify, cost
propagation) assumes it can walk nodes in topological order.

When a cycle is detected, the error message identifies the node where
the back-edge was found and the path of nodes involved.
"""

from __future__ import annotations

from ..ir_nodes import IRProgram, IRNode, CycleError


class CycleCheckResult:
    """Diagnostic result from cycle detection."""

    def __init__(self, order: list[tuple[str, IRNode]], is_dag: bool,
                 cycle_node: str | None = None):
        self.order = order
        self.is_dag = is_dag
        self.cycle_node = cycle_node
        self.node_count = len(order)

    @property
    def depth(self) -> int:
        """Longest path length in the DAG (critical path)."""
        if not self.order:
            return 0
        depths: dict[str, int] = {}
        for name, node in self.order:
            parent_depths = [
                depths.get(p.name, 0) for p in node.parents
                if p.name in depths
            ]
            depths[name] = (max(parent_depths) + 1) if parent_depths else 0
        return max(depths.values(), default=0)

    def __repr__(self) -> str:
        status = "acyclic" if self.is_dag else f"CYCLE at '{self.cycle_node}'"
        return (
            f"CycleCheckResult({status}, "
            f"nodes={self.node_count}, depth={self.depth})"
        )


def check_cycles(prog: IRProgram) -> list[tuple[str, IRNode]]:
    """Run topological sort on the IR program.

    Returns the topological ordering if acyclic.
    Raises CycleError if a cycle is detected.
    """
    return prog.topo_order()


def check_cycles_diagnostic(prog: IRProgram) -> CycleCheckResult:
    """Run cycle detection with full diagnostic output.

    Returns a CycleCheckResult even on failure (is_dag=False),
    rather than raising. Useful for tooling and error reporting.
    """
    try:
        order = prog.topo_order()
        return CycleCheckResult(order=order, is_dag=True)
    except CycleError as e:
        msg = str(e)
        cycle_node = None
        if "'" in msg:
            cycle_node = msg.split("'")[1]
        return CycleCheckResult(order=[], is_dag=False, cycle_node=cycle_node)


def find_back_edges(prog: IRProgram) -> list[tuple[str, str]]:
    """Identify all back-edges in the IR graph.

    A back-edge (u -> v) exists when v is an ancestor of u in the DFS
    tree — i.e. v is currently on the recursion stack when u is visited.

    Returns a list of (from_node, to_node) pairs. Empty list means DAG.
    """
    visited: set[str] = set()
    in_stack: set[str] = set()
    back_edges: list[tuple[str, str]] = []

    def visit(name: str):
        if name in visited:
            return
        visited.add(name)
        in_stack.add(name)

        node = prog._nodes.get(name)
        if node is not None:
            for parent in node.parents:
                if parent.name in in_stack:
                    back_edges.append((name, parent.name))
                elif parent.name not in visited and parent.name in prog._nodes:
                    visit(parent.name)

        in_stack.discard(name)

    for name in prog._nodes:
        visit(name)

    return back_edges

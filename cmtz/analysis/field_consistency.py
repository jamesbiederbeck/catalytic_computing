"""Field consistency checker for Cook-Mertz v2 IR.

Every IR node must have the same field as its parents.
Mixing fields silently produces wrong results because
modular reductions alias differently in different primes.
"""

from __future__ import annotations

from ..ir_nodes import IRProgram, IRNode


class FieldMismatchError(Exception):
    pass


def check_field_consistency(prog: IRProgram) -> None:
    """Walk the IR DAG topologically and verify field annotations match.

    Raises FieldMismatchError if any node has a parent with a
    different field annotation.
    """
    for name, node in prog.topo_order():
        if node.ir_field is None:
            continue
        for parent in node.parents:
            if parent.ir_field is None:
                continue
            if parent.ir_field != node.ir_field:
                raise FieldMismatchError(
                    f"Node '{name}' (field={node.ir_field}) has parent "
                    f"'{parent.name}' (field={parent.ir_field})"
                )

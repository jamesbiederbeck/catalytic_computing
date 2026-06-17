"""Field consistency checker for Cook-Mertz v4 IR.

Every IR node must have the same field as its parents, with one exception:
IRMagnitudeSq type-lowers F_{p²} → F_p and is the only cross-type boundary
allowed by the type rules in §6.5.

Additionally:
  - IRConjugate and IRComplexEmbed require ir_field.is_complex == True
  - Mixing real and complex parents at IRAdd is an error
"""

from __future__ import annotations

from ..ir_nodes import (
    IRProgram, IRNode, IRMagnitudeSq, IRConjugate, IRComplexEmbed,
)


class FieldMismatchError(Exception):
    pass


def check_field_consistency(prog: IRProgram) -> None:
    """Walk the IR DAG topologically and verify field annotations match.

    Raises FieldMismatchError for:
      - A node with a parent in a different field (same-field rule)
      - IRMagnitudeSq input not in a complex field
      - IRConjugate / IRComplexEmbed with a non-complex field
    """
    for name, node in prog.topo_order():
        if node.ir_field is None:
            continue

        # IRMagnitudeSq: cross-type boundary — check input is complex
        if isinstance(node, IRMagnitudeSq):
            if node.parents:
                parent = node.parents[0]
                if parent.ir_field is None:
                    continue
                if not parent.ir_field.is_complex:
                    raise FieldMismatchError(
                        f"IRMagnitudeSq '{name}' requires a complex input, "
                        f"but parent '{parent.name}' has real field {parent.ir_field}"
                    )
                # Output field should be the real counterpart
                expected_out = parent.ir_field.real_field()
                if node.ir_field != expected_out:
                    raise FieldMismatchError(
                        f"IRMagnitudeSq '{name}' output field {node.ir_field} "
                        f"does not match expected real field {expected_out}"
                    )
            continue  # skip generic parent-match check for this node

        # Complex-only ops
        if isinstance(node, (IRConjugate, IRComplexEmbed)):
            if not node.ir_field.is_complex:
                raise FieldMismatchError(
                    f"Node '{name}' ({type(node).__name__}) requires a complex "
                    f"field (cfield), but has {node.ir_field}"
                )

        # Generic same-field rule
        for parent in node.parents:
            if parent.ir_field is None:
                continue
            if parent.ir_field != node.ir_field:
                raise FieldMismatchError(
                    f"Node '{name}' (field={node.ir_field}) has parent "
                    f"'{parent.name}' (field={parent.ir_field})"
                )

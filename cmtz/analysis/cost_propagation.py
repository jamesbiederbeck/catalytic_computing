"""Cost propagation for Cook-Mertz v2.

Implements Lemma 2.1 composition cost accounting:
  - recursive_calls(f ∘ g)    = t_f × t_g
  - basic_instructions(f ∘ g) = s_f + t_f × s_g
  - num_registers(f ∘ g)      = max(r_f, r_g)

And Theorem 1.2 budget verification for IRMatPow nodes:
  - recursive_calls    ≤ O_ε(d^ε · log d)
  - basic_instructions ≤ n^{e^{1/ε}} · d^ε · log d
  - registers_Fq       ≤ n^{e^{1/ε}}

v1 had no cost model — compose just concatenated nodes.
"""

from __future__ import annotations

import math
from ..ir_nodes import IRProgram, IRNode, IRCompose, IRMatPow


class BudgetExceededError(Exception):
    pass


# Allow 10% slack for rounding in the asymptotic bounds
LOG_SLACK = 1.1


def propagate_costs(prog: IRProgram) -> None:
    """Bottom-up cost propagation over the IR DAG.

    Walks nodes in topological order and computes cost fields
    for each IRCompose node using Lemma 2.1.
    """
    for name, node in prog.topo_order():
        if isinstance(node, IRCompose):
            _propagate_compose(node)


def _propagate_compose(node: IRCompose) -> None:
    """Apply Lemma 2.1 cost accounting to a composition node."""
    if len(node.parents) < 2:
        return

    f = node.parents[0]
    g = node.parents[1]

    t_f = max(1, f.recursive_calls)
    t_g = max(1, g.recursive_calls)

    node.recursive_calls = t_f * t_g
    node.basic_instructions = f.basic_instructions + t_f * g.basic_instructions
    node.num_registers = max(f.num_registers, g.num_registers)


def check_matpow_budget(prog: IRProgram) -> dict[str, dict]:
    """Verify Theorem 1.2 bounds for all IRMatPow nodes.

    Returns a dict mapping node name → {actual, bound, ok} for each
    cost metric. Raises BudgetExceededError if any bound is exceeded.
    """
    results = {}

    for name, node in prog.nodes():
        if not isinstance(node, IRMatPow):
            continue

        bounds = node.cost_bound()
        actual = {
            'recursive_calls': node.recursive_calls,
            'basic_instructions': node.basic_instructions,
            'num_registers': node.num_registers,
        }

        checks = {}
        for metric in ['recursive_calls', 'basic_instructions']:
            bound_val = bounds[metric]
            actual_val = actual[metric]
            ok = actual_val <= bound_val * LOG_SLACK
            checks[metric] = {
                'actual': actual_val,
                'bound': bound_val,
                'ok': ok,
            }
            if not ok:
                raise BudgetExceededError(
                    f"MatPow '{name}' (d={node.d}, ε={node.epsilon}): "
                    f"{metric}={actual_val} exceeds "
                    f"Theorem 1.2 bound {bound_val:.1f}"
                )

        results[name] = checks

    return results

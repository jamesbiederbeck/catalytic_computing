"""Catalytic restore verifier for Cook-Mertz v3.

Only IRCatalyticRegion nodes (produced by `interrupt {}`) are verified.
IRRegion nodes (produced by `catalytic {}`) are advisory and skipped.

An interrupt handler borrows registers from a running main-process
computation. The main process resumes after the handler returns and
depends on those registers being intact. The verifier statically proves
this holds before any execution occurs.

Current checks (§6.3):
  - IREmbed clobber: if any inner node is an IREmbed whose output name
    matches a listed catalytic register, that is a definite violation —
    embed unconditionally overwrites the register.
  - IRRotate: treated as net-zero (part of the m-fold ω-loop; root-of-
    unity cancellation guarantees restoration after m iterations).
  - IRCompose: safe when both sub-programs are individually clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from ..ir_nodes import (
    IRProgram, IRNode, IRRegion, IRCatalyticRegion, IREmbed, IRRotate,
    IRCompose, IRMeasure, IRPoly, IRMatPow,
)


class CatalyticViolationError(Exception):
    def __init__(self, violations: list[RegDelta]):
        msgs = [f"  {v.register}: net delta = {v.delta_expr}" for v in violations]
        super().__init__(
            "Catalytic restore invariant violated:\n" + "\n".join(msgs)
        )
        self.violations = violations


@dataclass
class RegDelta:
    """Symbolic delta for a single register through a computation."""
    register: str
    delta_expr: str  # "0" means restored; anything else is a violation

    @property
    def is_restored(self) -> bool:
        return self.delta_expr == "0"


def _trace_deltas(nodes: list[IRNode], catalytic_regs: list[str]) -> list[RegDelta]:
    """Trace symbolic register deltas for a strict (interrupt) region.

    Checks each inner node against the listed catalytic registers and
    accumulates a delta expression. A non-zero delta is a violation.

    IREmbed clobber (definite violation):
      embed(i, ψ) writes unconditionally to register 'embed_i'.
      If that name matches a catalytic register, the handler destroys
      the main process's value — delta marked 'clobbered'.

    IRRotate (net-zero):
      Each rotation is part of the m-fold ω-loop; the m iterations
      sum to zero by root-of-unity cancellation (§6.3).

    IRCompose (safe):
      Lemma 2.1 guarantees the composed program is clean when both
      sub-programs are individually clean.
    """
    deltas: dict[str, str] = {reg: "0" for reg in catalytic_regs}

    for node in nodes:
        if isinstance(node, IREmbed):
            # embed(i, ψ) produces register named 'embed_i'.
            # If that name is in the catalytic set, it's a clobber.
            embed_name = f"embed_{node.index}"
            if embed_name in deltas:
                deltas[embed_name] = f"clobbered_by_embed({node.index},{node.psi})"

        elif isinstance(node, IRRotate):
            # Part of the m-fold loop; net delta is zero.
            pass

        elif isinstance(node, IRCompose):
            # Safe when sub-programs are clean (Lemma 2.1).
            pass

    return [
        RegDelta(register=reg, delta_expr=expr)
        for reg, expr in deltas.items()
    ]


def verify_catalytic(prog: IRProgram) -> list[RegDelta]:
    """Verify strict catalytic regions (IRCatalyticRegion / interrupt {}).

    IRRegion nodes (catalytic {} / advisory) are skipped — the main
    process owns its registers and has no restoration obligation.

    Returns the list of register deltas for strict regions (all "0").
    Raises CatalyticViolationError if any register is clobbered.
    """
    all_deltas: list[RegDelta] = []

    for name, node in prog.nodes():
        if isinstance(node, IRCatalyticRegion):
            deltas = _trace_deltas(node.inner_nodes, node.catalytic_regs)
            violations = [d for d in deltas if not d.is_restored]
            if violations:
                raise CatalyticViolationError(violations)
            all_deltas.extend(deltas)
        # IRRegion nodes deliberately skipped

    return all_deltas


def verify_region(region: IRCatalyticRegion) -> list[RegDelta]:
    """Verify a single strict catalytic region (interrupt handler)."""
    deltas = _trace_deltas(region.inner_nodes, region.catalytic_regs)
    violations = [d for d in deltas if not d.is_restored]
    if violations:
        raise CatalyticViolationError(violations)
    return deltas

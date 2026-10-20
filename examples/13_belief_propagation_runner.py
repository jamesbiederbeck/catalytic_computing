#!/usr/bin/env python3
"""
Run 13_belief_propagation.cmtz — a real-valued, single-frequency-bin sweep
of belief propagation over a 3-node chain X0 - X1 - X2, using the v5 mul()
primitive.

Background
----------
Belief propagation (BP) passes messages between nodes of a factor graph.
For circular/periodic variables (x in S^1), the messages can be represented
as Fourier coefficients, and BP's two operations become:

  factor-to-variable: multiply by the factor's Fourier coefficient phi_k
                       (known ahead of time -> compile-time constant)
  variable-to-factor: multiply together the incoming messages
                       (both computed at runtime -> two IR node outputs)

cmtz already had primitives for the first case (IRRotate/IRPoly multiply a
register by a compile-time constant) but nothing for the second: IRAdd (v4)
only adds, and nothing before v5 could multiply two *runtime-computed*
registers together. That's the gap IRMul closes.

What this example does and doesn't demonstrate
------------------------------------------------
- DOES: the variable-to-factor product step (mul()), which is the actual
  motivating gap, for a real-valued (F_p) chain, one fixed frequency bin,
  one full message-passing sweep, all hand-unrolled (the DSL has no loops).
- DOES NOT: the full complex/phasor Fourier-BP formulation from the design
  discussion (S^1-valued variables, cembed'd phasor messages, von-Mises-style
  kernels). That needs a *real* compile-time coefficient scaling a *complex*
  message, which routes through IRPoly — and IRPoly is deliberately
  real-only (see Cook_Mertz_Spec_v2.md §6.5's "IRPoly requires a real
  parent" note, added when we fixed the silent-garbage bug for complex
  IRPoly parents). Bridging that is separate, undecided scope.
- Every real coefficient used below happens to be representable as some
  omega^j (F_p* is cyclic — every nonzero element is *some* power of the
  primitive root), so the factor-to-variable step here uses rotate()
  rather than needing IRPoly at all.

Usage:
    cd into the repo root
    uv run python examples/13_belief_propagation_runner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cmtz import compile_program
from cmtz.field import primitive_roots_Fp

# ── Compile ───────────────────────────────────────────────────────────────────

source = (Path(__file__).parent / "13_belief_propagation.cmtz").read_text()
result = compile_program(source, backend='python', optimize=True)

if not result.ok:
    print("Compilation errors:")
    for e in result.errors:
        print(f"  {e}")
    sys.exit(1)

print("Compilation: OK")
print(f"  IR nodes      : {len(result.ir_program)}")
print(f"  acyclic       : {result.analysis_report['acyclic']}")
print(f"  field_consist : {result.analysis_report['field_consistent']}")
print(f"  catalytic     : {result.analysis_report['catalytic_verified']}")

# ── Show all register values ──────────────────────────────────────────────────

print()
print("Register values:")
out = result.backend_output
for name, val in out.items():
    print(f"  {name:20s} = {val}")

# ── Verify against a hand-derived reference computation ────────────────────────

p = 7
roots = primitive_roots_Fp(p)  # (1, 3, 2, 6, 4, 5)

evidence_X0 = roots[1]         # embed(0, 1) -> omega^1
evidence_X2 = roots[2]         # embed(1, 2) -> omega^2

phi01_k = roots[2]             # rotate(..., 2): factor(X0,X1) coefficient
phi21_k = roots[1]             # rotate(..., 1): factor(X2,X1) coefficient

msg01 = (evidence_X0 * phi01_k) % p     # factor-to-variable, X0 -> X1
msg21 = (evidence_X2 * phi21_k) % p     # factor-to-variable, X2 -> X1
belief_X1 = (msg01 * msg21) % p         # variable-to-factor product at X1

checks = [
    ("embed_0", out.get("embed_0"), evidence_X0),
    ("embed_1", out.get("embed_1"), evidence_X2),
    ("rot_embed_0_msg01", out.get("rot_embed_0_msg01"), msg01),
    ("rot_embed_1_msg21", out.get("rot_embed_1_msg21"), msg21),
    ("belief_X1", out.get("belief_X1"), belief_X1),
    ("result", out.get("result"), belief_X1),
]

print()
print("Verification:")
all_ok = True
for name, computed, expected in checks:
    ok = computed == expected
    all_ok = all_ok and ok
    flag = "✓" if ok else f"✗  (expected {expected})"
    print(f"  {name:20s} = {str(computed):5s}  {flag}")

print()
print(f"All correct: {all_ok}")
print()
print(f"X1's belief at this frequency bin: {belief_X1} "
      f"= ({evidence_X0}·{phi01_k}) · ({evidence_X2}·{phi21_k}) mod {p}")

sys.exit(0 if all_ok else 1)

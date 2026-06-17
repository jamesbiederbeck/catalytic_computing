#!/usr/bin/env python3
"""
Run 06_phasor_basics.cmtz — complex F_{p²} phasor network.

Demonstrates v4 operations: cfield, cembed, conj, magsq, add.
The Python backend handles complex registers as (re, im) tuples.

Usage:
    cd /home/victor/code/cook_mertz
    uv run python examples/06_phasor_runner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cmtz import compile_program

# ── Compile ───────────────────────────────────────────────────────────────────

source = (Path(__file__).parent / "06_phasor_basics.cmtz").read_text()
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
    if isinstance(val, tuple):
        re, im = val
        print(f"  {name:12s} = ({re}, {im})  [{re}+{im}i in F_7²]")
    else:
        print(f"  {name:12s} = {val}")

# ── Verify expected values ────────────────────────────────────────────────────

p = 7
c = 3    # quadratic non-residue mod 7

from cmtz.field import add_Fp2, conj_Fp2, magsq_Fp2

g = 3    # primitive root mod 7
roots7 = [pow(g, j, p) for j in range(p - 1)]

cembed_1_expected = (roots7[0], roots7[0])         # (1, 1)
cembed_2_expected = (roots7[1], roots7[2])         # (3, 2)
conj_2_expected   = conj_Fp2(cembed_2_expected, p) # (3, 5)
magsq_2_expected  = magsq_Fp2(cembed_2_expected, p, c)  # 4
sum_12_expected   = add_Fp2(cembed_1_expected, cembed_2_expected, p)  # (4, 3)

checks = [
    ("cembed_1",  out.get("cembed_1"),  cembed_1_expected),
    ("cembed_2",  out.get("cembed_2"),  cembed_2_expected),
    ("conj_2",    out.get("conj_2"),    conj_2_expected),
    ("magsq_2",   out.get("magsq_2"),   magsq_2_expected),
    ("sum_12",    out.get("sum_12"),     sum_12_expected),
    ("norm",      out.get("norm"),       magsq_2_expected),
]

print()
print("Verification:")
all_ok = True
for name, computed, expected in checks:
    ok = computed == expected
    all_ok = all_ok and ok
    flag = "✓" if ok else f"✗  (expected {expected})"
    print(f"  {name:12s} = {str(computed):12s}  {flag}")

print()
print(f"All correct: {all_ok}")

# ── Show the F_{p²} structure ────────────────────────────────────────────────

print(f"\nField structure: F_{{{p}²}} = F_{p}[i]/(i²-{c})")
print(f"  Non-residue check: {c}^{(p-1)//2} = {pow(c,(p-1)//2,p)} ≡ -1 mod {p}  ✓")
print(f"  magsq formula: a²-b²·c = 3²-2²·{c} = 9-12 = -3 ≡ {(-3)%p} mod {p}")
print(f"  conj(3+2i) = 3+(-2 mod {p})i = 3+{(-2)%p}i")

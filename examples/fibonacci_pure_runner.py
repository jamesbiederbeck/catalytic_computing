#!/usr/bin/env python3
"""
Run fibonacci_pure.cmtz — Fibonacci via unrolled add chain, no matpow.

This demonstrates that the Cook-Mertz DSL can express Fibonacci
computation using only embed + add, without any external matrix
or Python-side glue code.

Usage:
    cd /home/victor/code/cook_mertz
    uv run python examples/fibonacci_pure_runner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cmtz import compile_program

# ── Compile ───────────────────────────────────────────────────────────────────

source = (Path(__file__).parent / "fibonacci_pure.cmtz").read_text()
result = compile_program(source, backend='python', optimize=True)

if not result.ok:
    print("Compilation errors:")
    for e in result.errors:
        print(f"  {e}")
    sys.exit(1)

print("Compilation: OK")
print(f"  IR nodes       : {len(result.ir_program)}")
print(f"  acyclic        : {result.analysis_report['acyclic']}")
print(f"  field_consist  : {result.analysis_report['field_consistent']}")
print(f"  catalytic      : {result.analysis_report['catalytic_verified']}")
print(f"  rotations fused: {result.optimization_stats.get('rotations_fused', 0)}")

# ── Read result directly from backend output ──────────────────────────────────

p = 17
out = result.backend_output   # dict[str, int] — all named registers
computed = out['result']

print(f"\nRegister values (mod {p}):")
for name, val in out.items():
    print(f"  {name:12s} = {val}")

# ── Verify against naive recurrence ──────────────────────────────────────────

def fib_naive(k, mod):
    a, b = 0, 1
    for _ in range(k):
        a, b = b, (a + b) % mod
    return a

n = 12
expected = fib_naive(n, p)
status = "✓" if computed == expected else "✗ MISMATCH"
print(f"\nResult: F_{n} mod {p} = {computed}  (expected {expected})  {status}")

# ── Show the full sequence mod p for context ──────────────────────────────────

print(f"\nFibonacci sequence mod {p}:")
seq = [fib_naive(k, p) for k in range(1, 13)]
print("  " + "  ".join(f"F{k}={v}" for k, v in enumerate(seq, 1)))

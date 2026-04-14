#!/usr/bin/env python3
"""
Run fibonacci.cmtz through the Cook-Mertz compiler and evaluate it.

The DSL handles structure, cost verification, and static analysis.
The runner provides the actual Fibonacci matrix and extracts results.

Usage:
    cd /home/victor/code/cook_mertz
    python examples/fibonacci_runner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cmtz import compile_program
from cmtz.ir_nodes import IRMatPow
from cmtz.backends.python_ref import PythonBackend
from cmtz.field import IRField

# ── Load and compile the DSL source ──────────────────────────────────────────

source = (Path(__file__).parent / "fibonacci.cmtz").read_text()
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
print(f"  budget        : {result.analysis_report['budget_check']}")
print(f"  rotations fused: {result.optimization_stats['rotations_fused']}")

# ── Find the IRMatPow node ────────────────────────────────────────────────────

matpow_node = next(
    (n for _, n in result.ir_program.nodes() if isinstance(n, IRMatPow)),
    None,
)
if matpow_node is None:
    print("No IRMatPow node found")
    sys.exit(1)

print(f"\nMatPow node: {matpow_node.name!r}")
print(f"  d       = {matpow_node.d}")
print(f"  epsilon = {matpow_node.epsilon}")
print(f"  delta   = {matpow_node.delta}  (= 2^(3/ε))")
cost = matpow_node.cost_bound()
print(f"  Theorem 1.2 bounds:")
print(f"    recursive_calls ≤ {cost['recursive_calls']:.2f}")
print(f"    basic_instrs    ≤ {cost['basic_instructions']:.2f}")

# ── Fibonacci matrix [[1,1],[1,0]] ────────────────────────────────────────────

FIB = [[1, 1],
       [1, 0]]

p = 17
be = PythonBackend(IRField(p=p, q=19))
Fn = be.eval_matpow_matrix(matpow_node, FIB)

# F^n · [1, 0]ᵀ = [F_{n+1}, F_n]ᵀ
# First column of F^n is [F^n[0][0], F^n[1][0]] = [F_{n+1}, F_n]
n = matpow_node.d
F_n   = Fn[1][0]   # F_12
F_np1 = Fn[0][0]   # F_13

# Verify against naive computation
def fib_naive(k):
    a, b = 0, 1
    for _ in range(k):
        a, b = b, (a + b) % p
    return a

print(f"\nResult: F^{n} mod {p}")
print(f"  matrix = [[{Fn[0][0]}, {Fn[0][1]}], [{Fn[1][0]}, {Fn[1][1]}]]")
print(f"\n  F_{n}   mod {p} = {F_n}")
print(f"  F_{n+1} mod {p} = {F_np1}")

naive_n   = fib_naive(n)
naive_np1 = fib_naive(n + 1)
print(f"\nNaive check:")
print(f"  F_{n}   mod {p} = {naive_n}  {'✓' if F_n == naive_n else '✗ MISMATCH'}")
print(f"  F_{n+1} mod {p} = {naive_np1}  {'✓' if F_np1 == naive_np1 else '✗ MISMATCH'}")

# ── Show several Fibonacci values via matpow ──────────────────────────────────

print(f"\nFibonacci sequence mod {p} via matpow:")
from cmtz.ir_nodes import IRMatPow as _IMP
from cmtz.field import IRField as _IF

field = IRField(p=p, q=19)
be2 = PythonBackend(field)
header = "  n  | F_n mod 17 | true F_n"
print(header)
print("  " + "-" * (len(header) - 2))

true_fibs = [0, 1]
for _ in range(18):
    true_fibs.append(true_fibs[-1] + true_fibs[-2])

for d in [1, 2, 3, 5, 8, 10, 12, 16, 18]:
    node = _IMP(name=f"tmp", ir_field=field, matrix_name="F", d=d, epsilon=0.5, n_dim=2)
    mat = be2.eval_matpow_matrix(node, FIB)
    val = mat[1][0]
    print(f"  {d:2d} | {val:10d} | {true_fibs[d]} (≡ {true_fibs[d] % p} mod {p})")

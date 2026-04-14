#!/usr/bin/env python3
"""
Demonstrates what "arbitrary Fibonacci" means for the Cook-Mertz compiler.

Three questions answered:
  1. Can it compute F_n for large n?      Yes — matpow scales in O(log_δ n) steps.
  2. Can n be a runtime variable?         No — d is a compile-time DSL literal.
  3. Workaround: programmatic DSL gen?    Yes — generate source strings from Python.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cmtz import compile_program
from cmtz.ir_nodes import IRMatPow
from cmtz.backends.python_ref import PythonBackend
from cmtz.field import IRField, to_base

FIB = [[1, 1], [1, 0]]          # Fibonacci recurrence matrix
p   = 1_000_003                 # large prime (avoids trivial mod-out)
q   = 1_000_033                 # next prime after p
field = IRField(p=p, q=q)
be  = PythonBackend(field)

def fib_matpow(n: int, epsilon: float = 0.5) -> int:
    """Compute F_n mod p via base-δ decomposition, no DSL round-trip."""
    node = IRMatPow(name="F", ir_field=field, matrix_name="fib_matrix",
                    d=n, epsilon=epsilon, n_dim=2)
    mat = be.eval_matpow_matrix(node, FIB)
    return mat[1][0]   # F^n · [1,0]ᵀ gives [F_{n+1}, F_n]ᵀ; take row 1

def fib_naive(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % p
    return a

# ── 1. Step count: how many matrix multiplies does base-δ need? ──────────────

print("=== Step count: digits of n in base δ=64 ===")
print(f"  {'n':>12}  digits  mat-muls")
for n in [12, 100, 1_000, 1_000_000, 10**9, 10**18]:
    digits = to_base(n, 64)
    # each digit requires: 1 mat-pow-small (the block) + 1 advance step
    mulmuls = len(digits) * 2
    print(f"  {n:>12,}  {len(digits):>6}  ~{mulmuls}")

# ── 2. Correctness check for small n ─────────────────────────────────────────

print("\n=== Correctness check against naive (mod 1_000_003) ===")
ok = True
for n in [1, 2, 5, 10, 50, 100, 500]:
    mp = fib_matpow(n)
    nv = fib_naive(n)
    flag = "✓" if mp == nv else "✗"
    print(f"  F_{n:<4} = {mp:<12}  naive={nv:<12} {flag}")
    ok = ok and (mp == nv)
print(f"  All correct: {ok}")

# ── 3. Large n: what can't be done naively ────────────────────────────────────

print("\n=== Large n: matpow vs naive timing ===")
large_cases = [
    (10_000,     "10k"),
    (1_000_000,  "1M"),
    (100_000_000,"100M"),
]
for n, label in large_cases:
    t0 = time.perf_counter()
    val = fib_matpow(n)
    t_mp = time.perf_counter() - t0

    if n <= 100_000:
        t0 = time.perf_counter()
        nv = fib_naive(n)
        t_nv = time.perf_counter() - t0
        match = "✓" if val == nv else "✗"
        print(f"  F_{label:<5}: matpow={val}  ({t_mp*1000:.2f}ms)  naive={t_nv*1000:.1f}ms  {match}")
    else:
        print(f"  F_{label:<5}: matpow={val}  ({t_mp*1000:.2f}ms)  naive=too slow")

# ── 4. The compile-time constraint ────────────────────────────────────────────

print("\n=== The compile-time constraint ===")
print("  In the DSL, d must be a literal integer:")
print("    matpow(fib_matrix, 12, 0.5) as F12;   -- OK")
print("    matpow(fib_matrix, n, 0.5)  as Fn;    -- PARSE ERROR: n is not INT")
print()
print("  Workaround: generate DSL source from Python for any n.")

def fib_dsl_source(n: int, prime_p: int = 17, prime_q: int = 19,
                   epsilon: float = 0.5) -> str:
    return f"""\
field({prime_p}, {prime_q});
roots({prime_p}) as fib_matrix;
embed(0, 0);
catalytic {{
    matpow(fib_matrix, {n}, {epsilon}) as F{n};
}} restoring (r0, r1, r2, r3);
measure(F{n}, mod_p) as result;
"""

for n in [12, 144, 1597]:   # Fibonacci numbers themselves
    src = fib_dsl_source(n)
    result = compile_program(src, backend='python')
    if result.ok:
        node = next(nd for _, nd in result.ir_program.nodes()
                    if isinstance(nd, IRMatPow))
        be17 = PythonBackend(IRField(p=17, q=19))
        mat  = be17.eval_matpow_matrix(node, FIB)
        val  = mat[1][0]
        cost = node.cost_bound()
        print(f"  F_{n:<5}: {val} mod 17  "
              f"(budget: ≤{cost['recursive_calls']:.1f} recursive calls, "
              f"{len(to_base(n, node.delta))} base-{node.delta} digits)")
    else:
        print(f"  F_{n}: compile error: {result.errors}")

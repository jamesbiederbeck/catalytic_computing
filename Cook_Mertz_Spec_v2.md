# Cook–Mertz Computation Model — Specification v2

## 1. Introduction

The Cook–Mertz model defines a **branch-free**, arithmetic-circuit-based computation paradigm
for evaluating tree-structured programs in small space via catalytic register techniques.
This specification describes the **v2 revision** of the compiler and IR, which fixes three
foundational algebra errors in v1 and adds the missing invariants required for correctness.

The model implements the key insight from Cook & Mertz [CM24] and Goldreich's exposition [Gol24]:
polynomial interpolation via roots of unity in a prime field F_p enables space-efficient tree
evaluation with catalytic (restorable) workspace. The Alekseev–Cleve construction [AC26]
extends this with a catalytic pebbling method that achieves polynomial runtime for
TreeEval in O(log^{1+ε} n · log log n) space.

### What v2 fixes

| # | v1 Bug | Impact | v2 Fix |
|---|--------|--------|--------|
| 1 | `cyclo(n)` called `sympy.cyclotomic_poly()` | Computed Φ_n(x) (minimal polynomial over Q), not field elements | `roots(p)` enumerates {ω^j mod p : j ∈ [p-1]} via `primitive_root(p)` |
| 2 | `IRRotate(theta: float)` | Ambiguous: complex phase vs modular integer | `IRRotate(j: int, field)` — exact modular arithmetic |
| 3 | `IRPoly(coeffs)` with no field | Silent overflow via float64 | `IRPoly(coeffs, field)` — bigint coefficients, explicit mod q |
| 4 | No catalytic tracking | Composition correctness unverifiable | `IRCatalyticRegion` + symbolic restore verifier |
| 5 | No cost model | Cannot verify Theorem 1.2 bounds | Lemma 2.1 propagation + budget checker |
| 6 | `iimport torch`, mutable defaults, TOKEN_SPEC ordering | Runtime crashes, silent data sharing, parse failures | All fixed |


## 2. Core Primitives

### 2.1 Vertex Embedding (Phase Register)

- **Operation**: Initialize register with ω^ψ in F_p
- **Parameters**: index i, integer phase exponent ψ (v2: integer, not float)
- **IR Node**: `IREmbed(i: int, ψ: int, field: IRField)`

### 2.2 Edge Rotation (ω-gate)

- **Operation**: Multiply register value by ω^j in F_p
- **Parameters**: source register, integer exponent j (v2: not float θ)
- **Physical mapping**: θ_physical = 2π·j/m where m = p − 1
- **IR Node**: `IRRotate(j: int, field: IRField)`
- **Key property**: m consecutive rotations (j = 0..m−1) achieve root-of-unity
  cancellation: Σ_{j∈[m]} ω^j = 0 mod p. This is the mechanism underlying
  Lemma 4 [Gol24] — the polynomial interpolation identity.

### 2.3 Primitive Root Table

- **Operation**: Compute {ω^0, ω^1, ..., ω^(m−1)} in F_p where ω = primitive_root(p)
- **Replaces**: v1's `cyclo(n)` which computed the cyclotomic polynomial Φ_n(x)
- **IR Node**: `IRPrimitiveRoots(order: int, roots: tuple[int])`
- **Correctness**: F_p* is cyclic of order m = p−1; the generator ω generates all
  non-zero elements. These are the actual field values needed for the ω-gate.

### 2.4 Polynomial Evaluation

- **Operation**: Evaluate polynomial P(x) over F_q, reduce to F_p
- **Coefficients**: Exact Python integers (arbitrary precision)
- **Working field**: F_q where q = O(2^n · p^(d+1)) per Corollary 3.4.2
- **IR Node**: `IRPoly(coeffs: list[int], degree: int, field: IRField)`

### 2.5 Matrix Powering

- **Operation**: Compute M^d via base-δ decomposition (Lemma 3.12)
- **Parameters**: matrix M, exponent d, tradeoff parameter ε
- **Cost bounds** (Theorem 1.2): O_ε(d^ε · log d) recursive calls,
  n^{e^{1/ε}} · d^ε · log d basic instructions
- **IR Node**: `IRMatPow(d, epsilon, field: IRField)`

### 2.6 Catalytic Region

- **Operation**: Wrap a subgraph guaranteeing catalytic register restoration
- **Invariant**: All registers in `catalytic_regs` have net-zero delta after
  the inner computation (Definition 2.5, "clean register program")
- **IR Node**: `IRCatalyticRegion(inner_nodes, catalytic_regs: list[str])`

### 2.7 Composition

- **Operation**: Compose two sub-programs with Lemma 2.1 cost accounting
- **Cost rules**:
  - recursive_calls(f ∘ g) = t_f × t_g
  - basic_instructions(f ∘ g) = s_f + t_f · s_g
  - num_registers(f ∘ g) = max(r_f, r_g)
- **IR Node**: `IRCompose(parents=[f, g])`

### 2.8 Measurement / Read-out

- **Operation**: Extract scalar from register
- **Modes**: `real | imag | mag | trace | mod_p` (v2 adds `mod_p`)
- **IR Node**: `IRMeasure(mode, src)`


## 3. Field Arithmetic

### 3.1 IRField

Every IR node carries an `IRField(p, q)`:
- **p**: prime characteristic. Output values live in F_p = Z/pZ.
- **q**: working field prime. q ≥ p. Intermediate computation happens mod q to prevent
  overflow. For degree-d polynomials over n-bit inputs, Corollary 3.4.2 requires
  q ≥ 2^n · p^(d+1).
- **m = p − 1**: order of F_p*, the root-of-unity order.

### 3.2 Primitive Root Computation

```
primitive_roots_Fp(p):
    g ← smallest primitive root mod p  (via sympy.primitive_root)
    return (g^0 mod p, g^1 mod p, ..., g^(m-1) mod p)
```

Properties verified by the test suite:
- The table is a permutation of {1, 2, ..., p−1}
- roots[0] = 1 (ω^0 = 1)
- Σ roots[j] ≡ 0 (mod p)  — root-of-unity cancellation
- roots[j] · roots[k] ≡ roots[(j+k) mod m] (mod p)  — group structure
- g^m ≡ 1 (mod p)  — periodicity

### 3.3 Horner Evaluation

Polynomial evaluation uses Horner's method with mod q at every step:

```
horner_eval_Fq(coeffs, x, q):
    result ← 0
    for c in reverse(coeffs):
        result ← (result · x + c) mod q
    return result
```

This prevents intermediate values from exceeding q^2, keeping them
representable in 64-bit integers for q < 2^31.


## 4. DSL v2 Syntax

```bnf
program      ::= field_decl? statement*

field_decl   ::= 'field' '(' p=INT ',' q=INT ')' ';'
               | 'field' '(' p=INT ')' ';'          -- q defaults to p

statement    ::= embed_stmt | rotate_stmt | compose_stmt
               | measure_stmt | roots_stmt | cyclo_phi_stmt
               | matpow_stmt | catalytic_stmt

roots_stmt   ::= 'roots' '(' p=INT ')' 'as' ID ';'
cyclo_phi_stmt ::= 'cyclo_phi' '(' n=INT ')' 'as' ID ';'
rotate_stmt  ::= 'rotate' '(' src=ID ',' dst=ID ',' j=INT ')' ';'
matpow_stmt  ::= 'matpow' '(' M=ID ',' d=INT ',' epsilon=FLOAT ')' 'as' ID ';'
catalytic_stmt ::= 'catalytic' '{' statement* '}' 'restoring' '(' ID_LIST ')' ';'
measure_stmt ::= 'measure' '(' id=ID ',' mode=MODE ')' 'as' var=ID ';'
MODE         ::= 'real' | 'imag' | 'mag' | 'trace' | 'mod_p'
```

### Example: M^8 with ε=0.5

```cmtz
field(17, 289);          -- p=17 (prime), q=17^2=289
roots(17) as omega17;    -- {ω^j mod 17 : j ∈ [16]}

embed(0, 0);
embed(1, 4);

catalytic {
    matpow(M, 8, 0.5) as Mpow8;
} restoring (r0, r1, r2, r3);

measure(Mpow8, mod_p) as result;
```


## 5. Compilation Pipeline

```
DSL source (.cmtz)
       │
       ▼
  [Lexer]  — token stream with line/col for error reporting
       │     MODE and keywords match before ID in alternation
       ▼
  [Parser] — recursive descent, produces typed AST
       │     rejects unknown IDs, enforces literal params
       ▼
  [Elaborator] — resolves field(p,q), validates primality,
       │          checks roots(p) consistency with field decl
       ▼
  [IR Lowering] — AST → IR DAG with field annotations
       │          roots(p) → IRPrimitiveRoots.build()
       │          rotate(i,j) → IRRotate with j mod m
       │          catalytic{} → IRCatalyticRegion
       │          compose → IRCompose with Lemma 2.1 costs
       ▼
  [Static Analysis]
       ├── Cycle check (topological sort)
       ├── Field consistency (all operands share same field)
       ├── Catalytic restore check (symbolic delta tracking)
       └── Cost propagation (Lemma 2.1) + Thm 1.2 budget check
       │
       ▼
  [IR Optimization]
       ├── Rotate fusion: ω^j1 · ω^j2 → ω^(j1+j2 mod m)
       ├── CSE on IRPoly nodes (identical coeffs + field)
       └── Constant folding (Horner eval at compile time)
       │
       ▼
  [Backend] (pluggable)
       ├── Python/numpy — exact Z_p arithmetic (reference)
       ├── PyTorch — int64 tensors, explicit % p every op
       └── Analog descriptor — JSON for hardware targeting
```


## 6. Static Analysis Passes

### 6.1 Cycle Check

Topological sort of the IR DAG. Failure indicates recursion that the
branch-free model cannot express. This is the first analysis pass because
all subsequent passes assume DAG structure.

### 6.2 Field Consistency

Every IR node's field must match its parents' field. Mixing primes silently
produces wrong results because modular reductions alias differently.

### 6.3 Catalytic Restore Verification

For each `IRCatalyticRegion`, verify that all catalytic registers have
net-zero symbolic delta. The Cook-Mertz pattern guarantees this via
the add/subtract pairing:
  τ ← τ + X_u  (add subtree value)
  ... F_u evaluation ...
  τ ← τ − X_u  (restore)

The m iterations of the ω-loop achieve polynomial interpolation
through root-of-unity cancellation while preserving the catalytic invariant.

### 6.4 Cost Propagation

Bottom-up propagation implementing Lemma 2.1. For each `IRCompose(f, g)`:
  - recursive_calls = t_f × t_g
  - basic_instructions = s_f + t_f · s_g
  - num_registers = max(r_f, r_g)

After propagation, every `IRMatPow` node is checked against its
Theorem 1.2 cost bound. Exceeding the bound (with 10% slack for
rounding) raises `BudgetExceededError`.


## 7. Backends

### 7.1 Python Reference Backend

Exact modular arithmetic over Z_p. No floating point anywhere.
Ground truth for correctness testing. Matrix powering uses
base-δ decomposition with repeated squaring mod p.

### 7.2 PyTorch Backend

Uses `torch.int64` tensors with explicit `% p` after every
arithmetic operation. Safe for p < 2^31. The root table is
precomputed as an int64 tensor on the target device.

v1 bugs fixed: `iimport torch` → proper import with availability
check; float64 → int64 throughout.

### 7.3 Analog Descriptor Backend

Emits a JSON descriptor mapping compiled programs to analog
hardware parameters: phase offsets θ_j = 2π·j/m, GEMM pass
count (= recursive_calls), register depth, tile size, and
restoration protocol hint.


## 8. Migration from v1

| v1 construct | v2 replacement | Why |
|---|---|---|
| `cyclo(n) as x` | `roots(p) as x` | Computes primitive roots, not Φ_n |
| `cyclo_poly` for filters | `cyclo_phi(n) as x` | Renamed to avoid confusion |
| `IRRotate(theta: float)` | `IRRotate(j: int, field)` | Stay in modular arithmetic |
| `IRPoly(coeffs: List[int])` | `IRPoly(coeffs, field, degree)` | Field-tagged, bigint safe |
| `iimport torch` | `import torch` | Typo fix |
| `IRNode.parents = []` | `field(default_factory=list)` | Mutable default fix |
| TOKEN_SPEC (MODE after ID) | MODE, keywords before ID | Tokenizer correctness |
| No catalytic tracking | `IRCatalyticRegion` + verifier | Core correctness |
| No cost model | Lemma 2.1 propagation | Theorem 1.2 verification |
| `compose` as list concat | `IRCompose` with cost accounting | Algebraically correct |


## 9. References

- [CM24] Cook, Mertz. "Tree Evaluation is in Space O(log n · log log n)." STOC 2024.
- [Gol24] Goldreich. "On the Cook-Mertz Tree Evaluation Procedure." 2024.
- [AC26] Alekseev, Cleve. "TreeEval in quasi-polynomial time and O(log^{1+ε} n · log log n) space." 2026.
- [Wil25] Williams. "Every multi-tape TM running in time t can be simulated in space O(√t log t)." 2025.
- [BCK+14] Buhrman et al. "Computing with a Full Memory: Catalytic Space." STOC 2014.

*End of Cook–Mertz v2 Specification*

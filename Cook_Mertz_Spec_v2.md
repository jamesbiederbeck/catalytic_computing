# Cook–Mertz Computation Model — Specification v4

## 1. Introduction

The Cook–Mertz model defines a **branch-free**, arithmetic-circuit-based computation paradigm
for evaluating tree-structured programs in small space via catalytic register techniques.
This specification describes the **v4 revision**, which extends the computation model with
additive superposition, complex-valued registers over the extension field F_{p²}, and the
phasor network primitives (conjugate, magnitude squared) required to express interference
patterns in exact modular arithmetic.

The model implements the key insight from Cook & Mertz [CM24] and Goldreich's exposition [Gol24]:
polynomial interpolation via roots of unity in a prime field F_p enables space-efficient tree
evaluation with catalytic (restorable) workspace. The Alekseev–Cleve construction [AC26]
extends this with a catalytic pebbling method that achieves polynomial runtime for
TreeEval in O(log^{1+ε} n · log log n) space.

### What v2 fixed

| # | v1 Bug | Impact | v2 Fix |
|---|--------|--------|--------|
| 1 | `cyclo(n)` called `sympy.cyclotomic_poly()` | Computed Φ_n(x) (minimal polynomial over Q), not field elements | `roots(p)` enumerates {ω^j mod p : j ∈ [p-1]} via `primitive_root(p)` |
| 2 | `IRRotate(theta: float)` | Ambiguous: complex phase vs modular integer | `IRRotate(j: int, field)` — exact modular arithmetic |
| 3 | `IRPoly(coeffs)` with no field | Silent overflow via float64 | `IRPoly(coeffs, field)` — bigint coefficients, explicit mod q |
| 4 | No catalytic tracking | Composition correctness unverifiable | `IRCatalyticRegion` + symbolic restore verifier |
| 5 | No cost model | Cannot verify Theorem 1.2 bounds | Lemma 2.1 propagation + budget checker |
| 6 | `iimport torch`, mutable defaults, TOKEN_SPEC ordering | Runtime crashes, silent data sharing, parse failures | All fixed |

### What v3 changed

| # | v2 Limitation | v3 Change |
|---|---------------|-----------|
| 1 | All `catalytic {}` blocks were verified — main processes had same obligation as handlers | `catalytic {}` is now advisory (`IRRegion`); no verification |
| 2 | No distinction between owning and borrowing registers | Main processes own registers freely; interrupt handlers borrow and must restore |
| 3 | No interrupt handler construct | New `interrupt {}` DSL keyword produces `IRCatalyticRegion`; strictly verified |

### What v4 changes

| # | v3 Limitation | v4 Change |
|---|---------------|-----------|
| 1 | No addition — model was purely multiplicative | New `IRAdd` primitive; `add(a, b) as name` DSL form |
| 2 | Registers are single F_p elements — cannot represent phasors (complex amplitudes) | Complex extension field F_{p²} = F_p[i]/(i²−c); new `cfield(p)` declaration |
| 3 | No complex initialization | `IRComplexEmbed(re_psi, im_psi)` initializes ω^re + i·ω^im; `cembed` DSL form |
| 4 | No conjugation | `IRConjugate` computes a+bi → a−bi; `conj` DSL form |
| 5 | No magnitude extraction from complex registers | `IRMagnitudeSq` computes a²+b² mod p (type-lowers F_{p²} → F_p); `magsq` DSL form |
| 6 | Field consistency did not handle mixed real/complex programs | New type rules in §6.5: mixing F_p and F_{p²} parents is an error except at IRMagnitudeSq |

**Conceptual basis (v3).** Catalytic register programs are a strict subset of all programs
computable with roots-of-unity arithmetic over F_p. A main process owns its entire
register file and can compute any such program. An interrupt handler, by contrast,
runs concurrently alongside a main process and borrows some of its registers. The
borrowing is only safe if the handler provably restores every register it touches —
i.e., if the handler is catalytic. The v3 model enforces the invariant exactly where
ownership semantics require it, and nowhere else.

**Conceptual basis (v4).** The multiplicative structure of F_p* suffices for tree
evaluation via polynomial interpolation, but cannot express superposition. Phasor
networks — circuits where signals interfere constructively or destructively — require
additive gates. The v4 model adds addition and the complex extension F_{p²} so that
phasors (complex amplitudes) can be computed exactly in modular arithmetic. All
operations remain branch-free and integer; no floating point is introduced.


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

### 2.6 Advisory Region (main process)

- **Operation**: Annotate a subgraph as a logical group with named registers of interest
- **Invariant**: None enforced. The main process owns these registers and may compute
  any roots-of-unity program over them. The `restoring` clause is documentation only.
- **IR Node**: `IRRegion(inner_nodes, catalytic_regs: list[str])`
- **DSL keyword**: `catalytic`

### 2.7 Interrupt Handler (strict catalytic region)

- **Operation**: Wrap a subgraph that borrows registers from a running computation
- **Invariant**: All registers in `catalytic_regs` must have net-zero delta after the
  inner computation completes (Definition 2.5, "clean register program"). The verifier
  enforces this statically. Any `IREmbed` in the body whose name matches a listed
  register is a provable violation — embed is destructive.
- **IR Node**: `IRCatalyticRegion(inner_nodes, catalytic_regs: list[str])`
- **DSL keyword**: `interrupt`
- **Rationale**: An interrupt handler runs alongside the main process. The main process
  resumes after the handler returns and depends on its registers being intact. The
  catalytic invariant is the proof that resumption is safe.

### 2.8 Composition

- **Operation**: Compose two sub-programs with Lemma 2.1 cost accounting
- **Cost rules**:
  - recursive_calls(f ∘ g) = t_f × t_g
  - basic_instructions(f ∘ g) = s_f + t_f · s_g
  - num_registers(f ∘ g) = max(r_f, r_g)
- **IR Node**: `IRCompose(parents=[f, g])`

### 2.9 Measurement / Read-out

- **Operation**: Extract scalar from register
- **Modes**: `real | imag | mag | trace | mod_p` (v2 adds `mod_p`)
- **IR Node**: `IRMeasure(mode, src)`

### 2.10 Addition (Superposition Gate)

- **Operation**: Compute (a + b) mod p, or (a + b) mod p² for complex registers
- **Parameters**: two source registers of the same type (both F_p or both F_{p²})
- **IR Node**: `IRAdd(parents=[a, b], ir_field: IRField)`
- **Cost**: 1 basic instruction, 0 recursive calls (2 F_p additions for F_{p²} inputs)
- **DSL form**: `add(a, b) as name;`
- **Restriction**: both parents must share the same `IRField` (same p, q, and c). Mixing
  real and complex registers at `IRAdd` is a field consistency error.

### 2.11 Complex Register Initialization

- **Operation**: Initialize a complex register with the phasor ω^re_psi + i·ω^im_psi in F_{p²}
- **Parameters**: integer phase exponents re_psi, im_psi (both interpreted mod m = p−1)
- **IR Node**: `IRComplexEmbed(re_psi: int, im_psi: int, ir_field: IRField)`
- **Requires**: program must declare `cfield(p)` — `ir_field.is_complex` must be True
- **DSL form**: `cembed(re_psi, im_psi) as name;` — name defaults to `cembed_N`
- **Cost**: 1 basic instruction, 0 recursive calls

### 2.12 Complex Conjugate

- **Operation**: Compute the complex conjugate: a+bi → a−bi mod p
- **Parameters**: one source register in F_{p²}
- **IR Node**: `IRConjugate(ir_field: IRField)`
- **Requires**: `ir_field.is_complex` must be True; input must also be F_{p²}
- **DSL form**: `conj(a) as name;`
- **Cost**: 1 basic instruction (one negation mod p on the imaginary component)

### 2.13 Magnitude Squared (Type-Lowering)

- **Operation**: Compute |a+bi|² = a² + b² mod p, returning a real F_p element
- **Parameters**: one source register in F_{p²}
- **IR Node**: `IRMagnitudeSq(ir_field: IRField)`
- **Type rule**: input is F_{p²}, output is F_p — this is the only node that lowers type.
  The output `ir_field` must be the corresponding real field (same p and q, c=None).
- **DSL form**: `magsq(a) as name;`
- **Cost**: 3 basic instructions (two squares, one add, one mod p)
- **Physical interpretation**: measures the power of a phasor signal


## 3. Field Arithmetic

### 3.1 IRField

Every IR node carries an `IRField(p, q, c=None)`:
- **p**: prime characteristic. Output values live in F_p = Z/pZ.
- **q**: working field prime. q ≥ p. Intermediate computation happens mod q to prevent
  overflow. For degree-d polynomials over n-bit inputs, Corollary 3.4.2 requires
  q ≥ 2^n · p^(d+1).
- **m = p − 1**: order of F_p*, the root-of-unity order.
- **c** (optional): quadratic non-residue mod p. When set, registers in this field are
  elements of F_{p²} = F_p[i]/(i²−c). A non-residue satisfies c^((p−1)/2) ≡ −1 (mod p).
  `ir_field.is_complex` returns True iff c is not None.

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

### 3.4 F_{p²} Arithmetic

Elements of F_{p²} are pairs (a, b) representing a + b·i where i² ≡ c mod p.

```
add_Fp2((a,b), (c_,d))  = ((a+c_) % p,  (b+d) % p)
mul_Fp2((a,b), (c_,d))  = ((a*c_ + b*d*c) % p,  (a*d + b*c_) % p)
conj_Fp2((a,b))         = (a, (-b) % p)
magsq_Fp2((a,b))        = (a*a + b*b*c) % p      -- note: result is in F_p
```

The multiplication formula uses the Karatsuba-style identity for the quotient ring
F_p[i]/(i²−c): (a+bi)(c_+di) = (ac_ + bdc) + (ad + bc_)i.

`find_nonresidue(p)` returns the smallest integer c ∈ {2, 3, ...} with
c^((p−1)/2) ≡ −1 (mod p).

**Working field for complex programs.** The same Corollary 3.4.2 bounds apply —
the field q must accommodate intermediate F_{p²} multiplications. The practical
rule: use `cfield(p)` and let the elaborator compute q. If providing q explicitly
via `cfield(p, q)`, verify q ≥ p² (so that F_{p²} multiplications stay in range).


## 4. DSL v4 Syntax

```bnf
program        ::= field_decl? statement*

field_decl     ::= 'field'  '(' p=INT ',' q=INT ')' ';'
                 | 'field'  '(' p=INT ')' ';'                 -- q defaults to p
                 | 'cfield' '(' p=INT ')' ';'                 -- complex F_{p²}; q and c auto-computed
                 | 'cfield' '(' p=INT ',' q=INT ')' ';'       -- complex, explicit q; c auto-computed

statement      ::= embed_stmt | rotate_stmt | compose_stmt
                 | measure_stmt | roots_stmt | cyclo_phi_stmt
                 | matpow_stmt | catalytic_stmt | interrupt_stmt
                 | add_stmt | cembed_stmt | conj_stmt | magsq_stmt

roots_stmt     ::= 'roots' '(' p=INT ')' 'as' ID ';'
cyclo_phi_stmt ::= 'cyclo_phi' '(' n=INT ')' 'as' ID ';'
rotate_stmt    ::= 'rotate' '(' src=ID ',' dst=ID ',' j=INT ')' ';'
matpow_stmt    ::= 'matpow' '(' M=ID ',' d=INT ',' epsilon=FLOAT ')' 'as' ID ';'

catalytic_stmt ::= 'catalytic' '{' statement* '}' 'restoring' '(' ID_LIST ')' ';'
                   -- Advisory. Lowers to IRRegion. No verification.
                   -- Main-process use: documents register intent, not enforced.

interrupt_stmt ::= 'interrupt' '{' statement* '}' 'restoring' '(' ID_LIST ')' ';'
                   -- Strict. Lowers to IRCatalyticRegion. Verification enforced.
                   -- Handler use: listed registers must be provably restored.

add_stmt       ::= 'add' '(' a=ID ',' b=ID ')' 'as' ID ';'
                   -- Both operands must share the same IRField.

cembed_stmt    ::= 'cembed' '(' re_psi=INT ',' im_psi=INT ')' ';'
                   -- Requires cfield declaration. Name: cembed_N.

conj_stmt      ::= 'conj' '(' a=ID ')' 'as' ID ';'
                   -- Requires complex input (ir_field.is_complex).

magsq_stmt     ::= 'magsq' '(' a=ID ')' 'as' ID ';'
                   -- Requires complex input. Output is real (F_p).

measure_stmt   ::= 'measure' '(' id=ID ',' mode=MODE ')' 'as' var=ID ';'
MODE           ::= 'real' | 'imag' | 'mag' | 'trace' | 'mod_p'
```

### Example: main process with an interrupt handler

```cmtz
field(17, 19);
roots(17) as omega17;

-- Main process: owns embed_0 freely, no restoration obligation
embed(0, 3);
catalytic {
    matpow(omega17, 8, 0.5) as Mpow8;
} restoring (r0, r1);          -- advisory: documents r0, r1 as workspace

-- Interrupt handler: borrows r0 from the main process
-- Verifier enforces that r0 is not clobbered inside this block
interrupt {
    rotate(embed_0, embed_1, 4);
} restoring (r0);               -- strict: r0 must be provably intact on return

measure(Mpow8, mod_p) as result;
```

### Example: phasor network (v4)

```cmtz
-- Two phasors in F_{49} (7² = 49, but use F_{p²} with p=7, auto q and c)
cfield(7);

cembed(0, 1) as A;    -- A = ω^0 + i·ω^1 = 1 + i·g  (g = generator of F_7*)
cembed(2, 0) as B;    -- B = ω^2 + i·1

add(A, B) as C;       -- C = A + B  (superposition)
conj(A) as A_bar;     -- A_bar = conjugate of A

magsq(C) as power;    -- power = |C|² in F_7
measure(power, mod_p) as result;
```


## 5. Compilation Pipeline

```
DSL source (.cmtz)
       │
       ▼
  [Lexer]  — token stream with line/col for error reporting
       │     MODE and keywords match before ID in alternation
       │     'interrupt' keyword added in v3
       │     'cfield', 'add', 'cembed', 'conj', 'magsq' added in v4
       ▼
  [Parser] — recursive descent, produces typed AST
       │     rejects unknown IDs, enforces literal params
       ▼
  [Elaborator] — resolves field(p,q) / cfield(p), validates primality,
       │          computes non-residue c for cfield declarations,
       │          checks roots(p) consistency with field decl
       ▼
  [IR Lowering] — AST → IR DAG with field annotations
       │          roots(p)     → IRPrimitiveRoots.build()
       │          rotate(i,j)  → IRRotate with j mod m
       │          catalytic{}  → IRRegion          (advisory, v3)
       │          interrupt{}  → IRCatalyticRegion  (strict,   v3)
       │          compose      → IRCompose with Lemma 2.1 costs
       │          add(a,b)     → IRAdd              (v4)
       │          cembed(r,i)  → IRComplexEmbed     (v4)
       │          conj(a)      → IRConjugate        (v4)
       │          magsq(a)     → IRMagnitudeSq      (v4)
       ▼
  [Static Analysis]
       ├── Cycle check (topological sort)
       ├── Field consistency (all operands share same field; §6.5 for complex rules)
       ├── Catalytic restore check — IRCatalyticRegion only (v3: IRRegion skipped)
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

Runs on `IRCatalyticRegion` nodes only (`interrupt {}` blocks). `IRRegion`
nodes (`catalytic {}` blocks) are skipped — the main process owns its
registers and has no restoration obligation.

For each `IRCatalyticRegion`, the verifier checks that all listed registers
have net-zero symbolic delta after the inner computation. Current checks:

- **Destructive write (IREmbed)**: if any node inside the handler is an
  `IREmbed` whose output name matches a listed catalytic register, this is
  a definite violation — embed unconditionally overwrites the register.
- **Rotation (IRRotate)**: rotations on a catalytic register are part of
  the m-fold ω-loop pattern; their net effect after m iterations is zero
  by root-of-unity cancellation. Treated as net-zero symbolically.
- **Composition (IRCompose)**: safe if both sub-programs are clean
  (Lemma 2.1 guarantees this for correctly structured programs).

The Cook-Mertz add/subtract restore pattern remains the canonical proof
of compliance for more complex handlers:
  τ ← τ + X_u  (add subtree value)
  ... F_u evaluation ...
  τ ← τ − X_u  (restore)

### 6.5 Complex Field Consistency (v4)

Extends §6.2 with type rules for the F_{p²}/F_p boundary:

- **Same-field rule**: all binary operands (`IRAdd`, `IRRotate`) must share the same
  `IRField` exactly (same p, q, and c). Mixing F_p and F_{p²} at any binary op is an error.
- **Complex-only ops**: `IRConjugate` and `IRComplexEmbed` require `ir_field.is_complex`.
  Using them in a real-field program is a field consistency error.
- **Type-lowering at IRMagnitudeSq**: the single allowed cross-type boundary. Input field
  must have `is_complex=True`; output field is the corresponding real field (same p, q, c=None).
  The verifier checks that the output node uses the real field, not the complex one.
- **IRMeasure on complex registers**: allowed for `real` and `imag` modes (extracts
  individual components); not allowed for `mod_p` on a complex register directly —
  use `magsq` first.

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
base-δ decomposition with repeated squaring mod p. Both `IRRegion`
and `IRCatalyticRegion` nodes evaluate their inner nodes identically
at runtime — the distinction is purely a static analysis concern.

**v4 complex support**: Complex registers are represented as Python `(int, int)` tuples.
`IRAdd` on complex inputs uses `add_Fp2`; `IRConjugate` uses `conj_Fp2`;
`IRMagnitudeSq` uses `magsq_Fp2` and returns a plain `int` in F_p.
The `registers` dict holds either `int` (real) or `(int, int)` (complex) values —
dispatch is by `isinstance(val, tuple)`.

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

### 7.4 GLSL Compute Shader Backend

Emits a fully unrolled GLSL 4.50 compute shader. Branch-free integer
modular arithmetic; SSBO for root table.

**v4 complex support**: Complex registers are represented as `uvec2` (re, im pair).
New GLSL helpers `add_fp2`, `mul_fp2`, `conj_fp2`, `magsq_fp2` are emitted as
inlined functions alongside the existing `mul_mod_p`/`add_mod_p` helpers.
`IRMagnitudeSq` output is a `uint` (real), ending the `uvec2` chain.


## 8. Migration

### v1 → v2

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

### v2 → v3

| v2 construct | v3 equivalent | Behaviour change |
|---|---|---|
| `catalytic {} restoring (...)` | unchanged syntax | Now lowers to `IRRegion`; verification no longer runs |
| *(no equivalent)* | `interrupt {} restoring (...)` | New; lowers to `IRCatalyticRegion`; strictly verified |
| `IRCatalyticRegion` (all regions) | `IRRegion` (advisory) / `IRCatalyticRegion` (strict) | Type now carries semantic meaning |

Existing v2 programs using `catalytic {}` compile unchanged. They lose
verification (which was always passing for well-formed programs anyway),
and gain the ability to contain non-catalytic computation without error.
To opt back into strict verification, rename the relevant blocks to
`interrupt {}`.

### v3 → v4

| v3 construct | v4 equivalent | Notes |
|---|---|---|
| `field(p, q)` | unchanged | Real programs unchanged |
| *(no equivalent)* | `cfield(p)` | Declares complex F_{p²} program |
| `embed(i, psi)` | unchanged for real | Use `cembed(re, im)` for complex init |
| *(no equivalent)* | `add(a, b) as name` | Addition gate |
| *(no equivalent)* | `conj(a) as name` | Complex conjugate |
| *(no equivalent)* | `magsq(a) as name` | |a|² → F_p |
| `IRField(p, q)` | `IRField(p, q, c=None)` | c=None means real; backward compatible |

All v3 programs compile unchanged under v4. New keywords (`cfield`, `add`, `cembed`,
`conj`, `magsq`) are only needed for complex/additive programs.


## 9. References

- [CM24] Cook, Mertz. "Tree Evaluation is in Space O(log n · log log n)." STOC 2024.
- [Gol24] Goldreich. "On the Cook-Mertz Tree Evaluation Procedure." 2024.
- [AC26] Alekseev, Cleve. "TreeEval in quasi-polynomial time and O(log^{1+ε} n · log log n) space." 2026.
- [Wil25] Williams. "Every multi-tape TM running in time t can be simulated in space O(√t log t)." 2025.
- [BCK+14] Buhrman et al. "Computing with a Full Memory: Catalytic Space." STOC 2014.

*End of Cook–Mertz v3 Specification*

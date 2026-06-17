# RFC-002: `recur` — Linear Recurrence Node

**Status:** Draft  
**Affects:** Lexer, Parser, AST, Elaborator, Lowering, IR, Backends  

---

## Motivation

Many useful computations are linear recurrences over F_p:

```
x[n] = c_1·x[n-1] + c_2·x[n-2] + ... + c_k·x[n-k]
```

Fibonacci (k=2, coefficients [1,1]), tribonacci (k=3), LFSR sequences, and
linear-recurrent integer sequences all fit this pattern.

Today the only DSL path to compute x[n] is `matpow` with an externally supplied
companion matrix, or an O(n)-node unrolled add chain. Neither is ergonomic:
`matpow` requires Python glue; the unrolled chain grows with n and does not
generalize.

`recur` is a first-class DSL node that encodes the recurrence coefficients and
step count. The lowerer automatically builds the companion matrix and lowers to
`IRMatPow`. The programmer never writes a matrix.

---

## Proposed Syntax

```
recur_stmt ::= "recur" "(" seeds "," coeffs "," INT ["," FLOAT] ")" "as" ID ";"
seeds      ::= "[" INT ("," INT)* "]"
coeffs     ::= "[" INT ("," INT)* "]"
```

- **seeds**: initial values `[x[0], x[1], ..., x[k-1]]`, length k
- **coeffs**: recurrence coefficients `[c_1, c_2, ..., c_k]` (c_1 multiplies the
  most recent term), length k
- **INT**: number of steps n to advance
- **FLOAT** (optional): epsilon for Theorem 1.2 cost bound, default 0.5

The result register holds x[n].

### Examples

```
-- Fibonacci: x[n] = x[n-1] + x[n-2], seeds [1, 1], 12 steps → F_12
field(17, 19);
recur([1, 1], [1, 1], 12) as F12;
measure(F12, mod_p) as result;
```

```
-- Tribonacci mod 101: x[n] = x[n-1] + x[n-2] + x[n-3]
field(101, 103);
recur([0, 0, 1], [1, 1, 1], 20) as T20;
measure(T20, mod_p) as result;
```

```
-- LFSR with taps [1, 0, 0, 1] (x[n] = x[n-1] + x[n-4]) mod 31
field(31, 37);
recur([1, 0, 0, 0], [1, 0, 0, 1], 64, 0.25) as lfsr;
measure(lfsr, mod_p) as result;
```

---

## Design

### New AST node

```python
@dataclass
class RecurStmt(ASTNode):
    name: str
    seeds: list[int]      # length k, initial state
    coeffs: list[int]     # length k, c_1 .. c_k
    steps: int            # n
    epsilon: float = 0.5
```

### Lowering: recur → companion matrix → IRMatPow

The lowerer constructs the k×k companion matrix C from the coefficients:

```
C = | c_1  c_2  c_3  ...  c_k |
    |  1    0    0   ...   0  |
    |  0    1    0   ...   0  |
    |  ...                    |
    |  0    0    0   ...   0  |
```

and the initial state column vector `v = [x[k-1], x[k-2], ..., x[0]]ᵀ`.

Then `C^n · v` gives `[x[n+k-1], x[n+k-2], ..., x[n]]ᵀ`, so `result = (C^n · v)[k-1]`.

The lowerer emits:

1. An `IRMatrix` node (from RFC-001) holding C — or, if RFC-001 is not
   implemented, a synthetic `IRPrimitiveRoots`-like leaf that stores the matrix
   inline on the IR node itself.
2. An `IRMatPow` node referencing that matrix, with `d = steps` and
   `epsilon = epsilon`.
3. An `IRRecurExtract` node (or reuse `IRMeasure`) that applies `C^n` to the
   seed vector and returns the bottom entry.

The alternative — constructing `C^n · v` in a new `IRRecur` node without
splitting into `IRMatrix + IRMatPow` — is simpler but would duplicate the
Theorem 1.2 cost machinery already in `IRMatPow`. Prefer reuse.

### New IR node (minimal path without RFC-001)

If RFC-001 is not landed first, the lowerer can attach the companion matrix
directly to a new thin wrapper node:

```python
@dataclass
class IRRecur(IRNode):
    seeds: list[int]
    companion: list[list[int]]   # k×k, built by lowerer
    steps: int
    epsilon: float
```

`IRRecur` lowers further to `IRMatPow` during a second lowering pass, or the
backend handles it directly. This avoids the RFC-001 dependency but duplicates
the matrix-storage concern.

**Recommendation:** implement RFC-001 first; `recur` then becomes pure syntactic
sugar that the lowerer desugars to `matrix + matpow` with no new IR nodes.

### Elaboration

The elaborator validates:
1. `len(seeds) == len(coeffs)` — both must be length k.
2. `steps >= 1`.
3. `0.0 < epsilon <= 1.0`.
4. A field must be declared before `recur` (same requirement as `matpow`).

The elaborator does not validate that the recurrence is non-degenerate (i.e.,
that the characteristic polynomial has full degree). That is left to the user.

### Result extraction

`C^n · v` is a k×1 vector. The result register should hold x[n], which is the
last entry of the advanced state vector. The backend extracts `(C^n · v)[k-1]`
(0-indexed) after matrix-vector multiplication.

In the Python backend this is a one-liner after `eval_matpow_matrix`:

```python
# state vector: [x[k-1], x[k-2], ..., x[0]]
v = list(reversed(seeds))
Cn = self.eval_matpow_matrix(matpow_node, companion)
# multiply Cn × v, take last row → x[n]
result = sum(Cn[k-1][j] * v[j] for j in range(k)) % self.p
```

### Backends

| Backend | Change |
|---|---|
| `python_ref` | Matrix-vector multiply after `eval_matpow_matrix`; extract index k-1 |
| `glsl` | Emit companion matrix as `const uint C[k*k]`, unroll mat-vec multiply |
| `analog` | Add `recur_coeffs` and `seeds` fields to `AnalogDescriptor` |
| `torch` | `torch.tensor` companion + `torch.linalg.matrix_power` |

---

## Cost Accounting

`recur` inherits `IRMatPow` cost accounting unchanged. The companion matrix
dimension is k×k, so Theorem 1.2 bounds apply with `n_dim = k`. Users can
tune `epsilon` as with any `matpow` call.

---

## Interaction with Existing Features

- `recur` is purely additive syntax; no existing programs are affected.
- `fibonacci_pure.cmtz` (the O(n) unrolled version) remains valid and can serve
  as a regression reference: `recur([1,1],[1,1],12)` must produce the same
  result.
- `fibonacci.cmtz` (the `matpow` + external matrix version) also remains valid.
  All three approaches should agree on F_12 mod 17 = 8.

---

## Open Questions

1. **Should `recur` be sugar or a first-class IR node?** Sugar (desugars to
   `matrix + matpow`) is simpler and reuses existing cost machinery. A first-class
   `IRRecur` node could carry semantic information (recurrence coefficients) that
   enables future algebraic optimizations (e.g., combining two recurrences with
   the same characteristic polynomial). Recommend sugar for now.

2. **Seed representation.** Seeds are raw integers in the DSL. Should they be
   field elements (reduced mod p at parse time) or arbitrary integers (reduced
   at evaluation time)? Recommend reduction at evaluation time, consistent with
   matrix literal entries in RFC-001.

3. **Non-homogeneous recurrences** (x[n] = c_1·x[n-1] + ... + f(n)) are out of
   scope. The companion matrix trick only applies to homogeneous linear
   recurrences.

---

## Implementation Order

1. Land RFC-001 (matrix literals) to provide `IRMatrix` and the register lookup
   in `eval_matpow`.
2. Add `RecurStmt` AST node, parser, and elaborator validation.
3. Add `_lower_recur` in `lowering.py`: build companion matrix, emit `IRMatrix`
   + `IRMatPow` nodes, attach seed vector to the `IRMatPow` node or a thin
   wrapper for result extraction.
4. Update Python backend to extract the k-1 entry after mat-pow.
5. Update GLSL, analog, and torch backends.
6. Add tests: Fibonacci, tribonacci, LFSR; verify all agree with naive recurrence.

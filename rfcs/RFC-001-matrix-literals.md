# RFC-001: Matrix Literal Syntax

**Status:** Draft  
**Affects:** Lexer, Parser, AST, Elaborator, Lowering, Backends  

---

## Motivation

`matpow(M, d, ε)` already encodes matrix exponentiation with full Theorem 1.2
cost verification, but the matrix itself must be supplied by the host language.
This creates a split: the DSL verifies cost, while Python provides data. For
programs where the matrix is a small, fixed constant (companion matrices,
rotation matrices, permutation matrices), this split is unnecessary friction.

The canonical example is Fibonacci:

```
-- Today: matrix must come from Python
roots(17) as fib_matrix;
matpow(fib_matrix, 12, 0.5) as F12;
```

With matrix literals the program is fully self-contained:

```
matrix [[1, 1], [1, 0]] as M;
matpow(M, 12, 0.5) as F12;
```

---

## Proposed Syntax

```
matrix_decl  ::= "matrix" "[" row ("," row)* "]" "as" ID ";"
row          ::= "[" INT ("," INT)* "]"
```

All entries are integer literals. The matrix is square; the elaborator
enforces this. Entries are implicitly reduced mod p at evaluation time.

### Examples

```
-- 2×2 Fibonacci companion matrix
field(17, 19);
matrix [[1, 1], [1, 0]] as M;
matpow(M, 12, 0.5) as F12;
measure(F12, mod_p) as result;
```

```
-- 3×3 Leslie matrix (population dynamics mod p)
field(101, 103);
matrix [[0, 3, 2],
        [1, 0, 0],
        [0, 1, 0]] as L;
matpow(L, 20, 0.5) as Ln;
measure(Ln, mod_p) as result;
```

---

## Design

### New AST node

```python
@dataclass
class MatrixDecl(ASTNode):
    name: str           # register name
    rows: list[list[int]]
```

`Program.statements` already holds a heterogeneous list; `MatrixDecl` is added
to the dispatch table in both the parser and lowering.

### New IR node

```python
@dataclass
class IRMatrix(IRNode):
    rows: list[list[int]]   # n×n integer matrix, entries unreduced
```

`IRMatrix` is a leaf node (no parents). The Python backend stores it in
`self.registers` as a `list[list[int]]`; `IRMatPow` looks up its matrix by
name using `self.registers[node.matrix_name]` instead of requiring the caller
to pass it in.

### Changes to `IRMatPow` evaluation

Current `PythonBackend.eval_matpow_matrix(node, matrix)` takes the matrix as
an argument. With this RFC, the backend would first check `self.registers` for
an `IRMatrix` node with the matching name and fall back to the explicit argument
for backwards compatibility:

```python
def eval_matpow(self, node: IRMatPow) -> list[list[int]]:
    mat = self.registers.get(node.matrix_name)
    if mat is None:
        return 0   # unchanged: no matrix registered, caller must supply
    return self.eval_matpow_matrix(node, mat)
```

### Elaboration

The elaborator validates:
1. All rows have the same length (square matrix).
2. The declared matrix name does not conflict with existing nodes.
3. The matrix name is visible in scope when `matpow` references it.

No field-arithmetic validation is performed at elaboration time; entries are
raw integers reduced mod p at evaluation.

### Backends

| Backend | Change |
|---|---|
| `python_ref` | `eval_matpow` looks up matrix from registers; `eval_matpow_matrix` unchanged for external use |
| `glsl` | Emit matrix as a `const uint mat[n*n]` in the shader prologue |
| `analog` | Emit matrix entries in the `AnalogDescriptor` JSON |
| `torch` | Convert rows to `torch.tensor(..., dtype=torch.int64) % p` |

---

## Interaction with Existing Features

- `roots(p) as M` continues to work unchanged; it creates an `IRPrimitiveRoots`
  node which `matpow` resolves by name. `IRMatrix` is a second valid node type
  for the `matrix_name` lookup.
- The cost accounting in `IRMatPow` (`delta`, `cost_bound()`) is unchanged.
- `fibonacci_runner.py` continues to work because `eval_matpow_matrix` keeps
  its existing signature.

---

## What This RFC Does Not Cover

- Non-square matrices.
- Symbolic matrix entries (expressions, field elements, register references).
- Runtime-variable matrices (the matrix is always a compile-time constant).

---

## Migration

No breaking changes. Programs using the `roots(...) as M; matpow(M, ...)` pattern
continue to work. Matrix literals are purely additive syntax.

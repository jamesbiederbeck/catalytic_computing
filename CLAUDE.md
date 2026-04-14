# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Set up environment (first time)
uv venv
uv add sympy
uv add pytest --dev

# Run all tests
uv run pytest cmtz/tests/

# Run a single test file
uv run pytest cmtz/tests/test_compiler.py -v

# Run a single test
uv run pytest cmtz/tests/test_compiler.py::TestEndToEnd::test_simple_embed_measure -v

# Compile a .cmtz script and print full report
uv run python compile_script.py

# Run a specific example
uv run python examples/fibonacci_runner.py
```

**Always use `uv` for package management** — never `pip` directly.

## Architecture

The compiler is a classic pipeline: DSL source → IR DAG → analysis → optimization → backend. The entry point is `compile_program()` in `cmtz/compiler.py`, which orchestrates all phases and returns a `CompilationResult`.

### Pipeline phases (in order)

1. **Lexer** (`lexer.py`) — tokenizes source. Token ordering matters: `MODE` and keyword tokens must appear before `ID` in `TOKEN_SPEC` or identifiers like `real`, `field` parse as generic IDs.

2. **Parser** (`parser.py`) — recursive descent producing typed AST nodes from `ast_nodes.py`. All statement parameters are literals; no expressions or variables.

3. **Elaborator** (`elaborator.py`) — resolves `field(p, q)`, validates both `p` and `q` are prime with `q ≥ p`, infers field from `roots(p)` if no explicit declaration. Produces an `IRField` for the lowering phase.

4. **Lowering** (`lowering.py`) — AST → IR DAG. Key naming conventions produced here:
   - `embed(i, ψ)` → register named `embed_i`
   - `rotate(src, dst, j)` → register named `rot_{src}_{dst}`
   - `compose(a, b, ...)` → fresh names `compose_1`, `compose_2`, ...
   - `catalytic { } restoring ()` → fresh name `catalytic_N`
   - `matpow(M, d, ε) as name` → register named `name`; `M` must already exist in the IR by name

5. **Static analysis** (`analysis/`) — four independent passes that all run even if one fails:
   - `cycle_check.py` — topological sort; must pass before anything else is meaningful
   - `field_consistency.py` — every node's `ir_field` must match its parents'
   - `catalytic_verify.py` — symbolic delta tracking through `IRCatalyticRegion`; currently initializes all deltas to 0 (all regions pass)
   - `cost_propagation.py` — bottom-up Lemma 2.1 cost propagation; checks `IRMatPow` nodes against Theorem 1.2 bounds with 10% slack

6. **Optimization** (`optimization/`) — three passes:
   - `rotate_fusion.py` — merges adjacent `IRRotate` nodes into one when the parent has no fan-out; updates `j = (j1 + j2) % m`
   - `cse.py` — deduplicates `IRPoly` nodes with identical `(coeffs, p, q)` signatures
   - `constant_fold.py` — evaluates `IRPoly` at compile time when input is constant

7. **Backends** (`backends/`) — pluggable, selected by the `backend` parameter:
   - `python_ref.py` — `PythonBackend`: exact Z_p arithmetic, walks IR via `topo_order()`. `IRMatPow` evaluates to 0 unless `eval_matpow_matrix(node, matrix)` is called directly with a concrete matrix.
   - `torch_backend.py` — `int64` tensors, explicit `% p`; availability-gated
   - `analog_descriptor.py` — emits a JSON `AnalogDescriptor` with phase offsets, GEMM pass count, register depth for hardware targeting
   - `glsl_backend.py` — emits a fully unrolled GLSL 4.50 compute shader (branch-free, integer mod arithmetic); SSBO for root table; `spirv_compile_command()` returns the glslangValidator invocation

### Key data structures

- **`IRField(p, q)`** — frozen dataclass; `m = p - 1` is the root-of-unity order. Both `p` and `q` must be prime, `q ≥ p`. The elaborator enforces this; `IRField.__post_init__` also validates.
- **`IRProgram`** — `OrderedDict[str, IRNode]` maintaining insertion order; `topo_order()` does the topological sort. Raises `CycleError` if a cycle exists. Duplicate names raise `ValueError`.
- **`IRNode`** — base dataclass with `name`, `parents: list[IRNode]`, `ir_field`, and Lemma 2.1 cost fields (`recursive_calls`, `basic_instructions`, `num_registers`). **Mutable default `parents=[]` was a v1 bug — v2 uses `field(default_factory=list)`.**
- **`IRMatPow`** — has a `delta` property (`= 2^(3/ε)`) and `cost_bound()` returning Theorem 1.2 bounds. The `matrix_name` string is a reference to an existing IR node name.

### The `matpow` / matrix evaluation split

The DSL compiles structure and verifies cost bounds. Actual matrix computation requires calling `PythonBackend.eval_matpow_matrix(node, matrix)` directly in Python — the DSL backend returns 0 for `IRMatPow` nodes. The matrix (`list[list[int]]`) is not expressible in the DSL; it's a runtime parameter. See `examples/fibonacci_runner.py` for the pattern.

### Working field constraint

`q` must be a prime ≥ `p`. The spec example uses `field(17, 289)` but `289 = 17²` is not prime — this is illustrative only. Use `field(17, 293)` (the next prime above `17²`) in real programs. `compute_working_field(p, degree, n)` in `field.py` computes the correct `q` per Corollary 3.4.2.

### DSL register naming gotcha

`matpow(M, d, ε) as name` requires `M` to already be a named node in the IR. The idiomatic workaround is `roots(p) as M` which creates an `IRPrimitiveRoots` node with that name, satisfying the lookup.

## Spec

`Cook_Mertz_Spec_v2.md` is the authoritative reference. Section 4 has the complete BNF grammar. The spec documents six v1 bugs fixed in v2; the most important are: `cyclo(n)` replaced by `roots(p)` (primitive roots, not cyclotomic polynomial), `IRRotate.theta: float` replaced by `IRRotate.j: int`, and `IRPoly` now carries an `IRField`.
# cook-mertz

A compiler for the **Cook–Mertz computation model** — a branch-free, arithmetic-circuit DSL for evaluating tree-structured programs in small space using catalytic register techniques and polynomial interpolation over finite fields.

Based on Cook & Mertz [CM24] and Goldreich's exposition [Gol24], with the Alekseev–Cleve catalytic pebbling construction [AC26].

**v4** adds phasor networks: additive gates, complex registers over F_{p²}, conjugate, and magnitude squared — enabling exact modular interference arithmetic with no floating point.

## What it does

The compiler takes `.cmtz` source programs through a classic pipeline:

```
source → lexer → parser → elaborator → lowering → analysis → optimization → backend
```

Static analysis verifies field consistency, detects cycles, tracks catalytic register obligations, and checks Theorem 1.2 cost bounds. Backends include a Python reference evaluator, GLSL 4.50 compute shader emitter, and a WebAssembly (WAT) emitter.

## Quick start

```bash
uv venv
uv add sympy
uv run pytest cmtz/tests/        # run all tests
uv run python examples/fibonacci_pure_runner.py
uv run python examples/run_all.py
```

## Language

Programs declare a field, initialize registers, and chain operations. Comments use `--`.

```
-- Fibonacci F_12 mod 17 = 8, computed by unrolled addition chain
field(17, 19);

embed(0, 0);   -- embed_0 = ω^0 = 1  (F_1)
embed(1, 0);   -- embed_1 = ω^0 = 1  (F_2)

add(embed_0, embed_1) as F3;
add(F3, embed_1)      as F4;
add(F4, F3)           as F5;
-- ... continue to F12 ...

measure(F12, mod_p) as result;   -- expect 8
```

### Core operations

| Operation | Description |
|---|---|
| `field(p, q)` | Declare working field F_p with extension prime q |
| `embed(i, ψ)` | Initialize register to ω^ψ in F_p |
| `rotate(src, dst, j)` | Multiply by ω^j (exact modular) |
| `roots(p) as name` | Bind primitive root table for F_p |
| `compose(a, b, ...)` | Polynomial composition |
| `matpow(M, d, ε) as name` | Matrix power with Theorem 1.2 cost check |
| `add(a, b) as name` | Additive superposition (v4) |
| `measure(reg, mod_p) as name` | Read out register value |

### Catalytic regions

```
interrupt {
    embed(2, 3);
    rotate(embed_2, embed_2, 5);
} restoring (embed_1);
```

`interrupt {}` produces a **strictly verified** catalytic region — the verifier checks that listed registers are never clobbered by anything inside the body. `catalytic {}` is advisory (no enforcement), for main-process workspace documentation.

### Phasor networks (v4)

```
cfield(7);              -- F_7², c=3 (auto), q=53 (auto)

cembed(0, 0);           -- cembed_1 = 1 + 1i
cembed(1, 2);           -- cembed_2 = 3 + 2i

conj(cembed_2)  as conj_2;    -- 3 − 2i
magsq(cembed_2) as magsq_2;   -- a²−b²·c mod p (type-lowers F_{p²}→F_p)
add(cembed_1, cembed_2) as sum_12;

measure(magsq_2, mod_p) as norm;   -- expect 4
```

`cfield(p)` declares F_{p²} = F_p[i]/(i²−c) where c is the smallest quadratic non-residue mod p, computed automatically.

## Examples

| File | What it shows |
|---|---|
| `01_hello_field.cmtz` | Basic field, embed, rotate, measure |
| `02_rotate_fuse.cmtz` | Rotate-fusion optimization |
| `03_conjugate_catalytic.cmtz` | Advisory catalytic region |
| `04_add_superposition.cmtz` | Additive gates in F_p |
| `05_interrupt.cmtz` | Strict interrupt handler |
| `06_phasor_basics.cmtz` | F_{p²} cembed / conj / magsq |
| `07_phasor_interference.cmtz` | Two-phasor interference pattern |
| `08_ntt_butterfly.cmtz` | NTT butterfly in F_{p²} |
| `09_interrupt_handler.cmtz` | Interrupt handler with complex registers |
| `10_fermat_little.cmtz` | Fermat's little theorem verification |
| `11_diffie_hellman.cmtz` | DH key exchange over F_p |
| `12_cost_chain.cmtz` | Theorem 1.2 cost chain |
| `fibonacci_pure.cmtz` | F_12 mod 17 via pure addition chain |
| `fibonacci.cmtz` | Fibonacci via matpow |

## Backends

- **`python_ref`** — exact Z_p reference evaluator (default)
- **`glsl`** — emits a fully unrolled GLSL 4.50 compute shader; call `spirv_compile_command()` for the glslangValidator invocation
- **`wasm`** — emits WebAssembly Text Format (WAT)

## Visualizer

```bash
uv run python visualizer/server.py
```

Opens a browser app that compiles `.cmtz` source and renders the IR DAG.

## Field constraint

`q` must be a prime ≥ `p`. The spec uses `field(17, 289)` as an illustration, but 289 = 17² is not prime — use `field(17, 293)` (next prime above 17²) in real programs. `compute_working_field(p, degree, n)` in `field.py` computes the correct `q` per Corollary 3.4.2.

## RFCs

- `rfcs/RFC-001-matrix-literals.md` — inline matrix literals in the DSL
- `rfcs/RFC-002-recur.md` — recursive program definition

## References

- [CM24] Cook & Mertz — original catalytic computation model
- [Gol24] Goldreich — polynomial interpolation exposition
- [AC26] Alekseev & Cleve — catalytic pebbling, O(log^{1+ε} n · log log n) space

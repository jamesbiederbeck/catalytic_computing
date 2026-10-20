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

# Fast loop: skip the iverilog simulations (full suite ~2 min, this ~1 s)
uv run pytest cmtz/tests/ -k "not Simulation"

# Visualizer (IR DAG + Board panel) on http://localhost:8765; Node tests for its protocol client
uv run python visualizer/server.py
node --test visualizer/board.test.mjs
UPDATE_GOLDEN=1 uv run pytest cmtz/tests/test_hwview.py   # regenerate visualizer/testdata/golden.json

# FPGA: one apio project per lowerable example (add --debug for state readback)
uv run python examples/fpga/build_projects.py <dir> [--debug] [slug ...]
# Check a design that is loaded on the board (reads/writes the UART only)
uv run python examples/fpga/uart_check.py 08_ntt_butterfly.cmtz --basic

# cmtz-core (reusable processor): generate the project, then check it on the board
uv run python examples/fpga/build_projects.py <dir> --core
uv run python examples/fpga/core_run.py [--quick]
```

The Verilog simulation tests need `iverilog`/`vvp` on `PATH` and skip without them. The `tang-nano-20k` playground repo's `env.sh` puts them there along with apio's yosys, nextpnr and openFPGALoader (`source` it first).

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
   - `verilog_backend.py` — pipelined Verilog-2001 (one registered stage per runtime node, no FSM, no `%`) plus a self-checking testbench; `IRMatPow` is rejected. See **FPGA flow** below.
   - `isa_backend.py` — `emit_isa` (or `backend='isa'` with `backend_options`): a `ProgramImage` for **cmtz-core**, the reusable processor. Real field only; `matpow` runs (needs `matrix_dims`); `interrupt {}` becomes the interrupt handler. See **cmtz-core** below.

### Key data structures

- **`IRField(p, q)`** — frozen dataclass; `m = p - 1` is the root-of-unity order. Both `p` and `q` must be prime, `q ≥ p`. The elaborator enforces this; `IRField.__post_init__` also validates.
- **`IRProgram`** — `OrderedDict[str, IRNode]` maintaining insertion order; `topo_order()` does the topological sort. Raises `CycleError` if a cycle exists. Duplicate names raise `ValueError`.
- **`IRNode`** — base dataclass with `name`, `parents: list[IRNode]`, `ir_field`, and Lemma 2.1 cost fields (`recursive_calls`, `basic_instructions`, `num_registers`). **Mutable default `parents=[]` was a v1 bug — v2 uses `field(default_factory=list)`.**
- **`IRMatPow`** — has a `delta` property (`= 2^(3/ε)`) and `cost_bound()` returning Theorem 1.2 bounds. The `matrix_name` string is a reference to an existing IR node name.

### The `matpow` / matrix evaluation split

The DSL compiles structure and verifies cost bounds. Actual matrix computation requires calling `PythonBackend.eval_matpow_matrix(node, matrix)` directly in Python — the DSL backend returns 0 for `IRMatPow` nodes. The cmtz-core backend is the exception: it executes matpow, with the matrix written to data memory at run time (the node's scalar value is still 0). The matrix (`list[list[int]]`) is not expressible in the DSL; it's a runtime parameter. See `examples/fibonacci_runner.py` for the pattern.

### Working field constraint

`q` must be a prime ≥ `p`. The spec example uses `field(17, 289)` but `289 = 17²` is not prime — this is illustrative only. Use `field(17, 293)` (the next prime above `17²`) in real programs. `compute_working_field(p, degree, n)` in `field.py` computes the correct `q` per Corollary 3.4.2.

### DSL register naming gotcha

`matpow(M, d, ε) as name` requires `M` to already be a named node in the IR. The idiomatic workaround is `roots(p) as M` which creates an `IRPrimitiveRoots` node with that name, satisfying the lookup.

### FPGA flow

`emit_verilog` (`backends/verilog_backend.py`) → `emit_uart_wrapper` (`backends/verilog_uart.py`) → `examples/fpga/build_projects.py` (apio projects) and `hwview.py` (what `visualizer/` draws and the Board panel checks).

- **Only runtime values cost hardware.** Embeds, literal-`j` rotations and anything built only from constants fold at emit time through `PythonBackend`, so the hardware and the reference share one definition. A closed program is a constant; `inputs="all"` turns every embed an output depends on into a port (default: only leaves feeding an `IRMul`).
- **Expected values come from the reference.** Generated testbenches, `HardwareView.evaluate` and `uart_check.py` all use `evaluate_all`; it must not re-evaluate region bodies, which would clobber overridden inputs.
- **UART protocol** (`verilog_uart.py` docstring): one byte per lane (complex = re then im), all-`0xFF` reply on an out-of-range lane, stalled requests dropped after a timeout. A `debug=True` build adds a `dbg_state` bus and `I`/`R`/`S` commands (info frame with a program hash, run, state readback). The pushed `projects/cmtz/` use the basic protocol; `cmtz-mul` has a hand-written wrapper (`examples/fpga/uart_mul/`).
- **`visualizer/`**: `board.js` (protocol client, Web Serial and simulated transports, no DOM) is tested against `testdata/golden.json`, which `test_hwview.py` regenerates from the Python encoder, so the JS framing is checked against the real bytes.
- **Generated files** (`cmtz_*.v`, `*_tb.v`, `diagram.svg`, per-project READMEs) are not edited by hand; change the generator and rebuild. `examples/fpga/diagram.py` draws each project's `diagram.svg` from the hardware manifest: plain SVG (no `<style>`/classes/scripts/fonts/images, its own white background rect, `viewBox`/`width`/`height`, `<title>` and `aria-label`, label text on a white rect instead of a stroke halo) because GitHub's mobile app does not render Mermaid and draws SVGs as images that ignore dark mode; `test_fpga_projects.py` enforces those rules.

**Default to `-noalu`.** Build FPGA projects with `yosys-extra-options = -noalu` in `apio.ini` unless the default ALU mapping is needed (a design that does not fit in LUTs) or has been verified on the board. On the Tang Nano 20K (yosys 0.63, nextpnr-himbaechel, apicula) two `magsq` projects returned wrong answers on the chip with the default mapping (67 and 92 of 96 vectors) while RTL and post-synthesis simulation passed; with `-noalu` every vector matched. The toolchain cause is not isolated, so passing simulation proves nothing here. `ALU_VERIFIED` in `build_projects.py` lists the projects measured correct without the flag; a new example stays on `-noalu` until `uart_check.py` has matched it on the board both ways, and `ON_BOARD` (which controls the READMEs' "checked on the board" claim) is only extended after a real run. Load to SRAM first (volatile), not flash.

### cmtz-core

A fixed processor per field (`cmtz/isa/`) instead of a circuit per program. `encoding.py` is the single ISA definition (assembler, simulator and RTL all read it); `sim.py` is the reference the RTL testbenches compare against; `host.py` documents the UART protocol and drives a board; `demos.py` holds hand-written programs (catalytic interrupt loop, exhaustive per-instruction sweeps); `lib/matrix.s` is the matpow runtime. `backends/isa_core.py` and `isa_uart.py` generate the core, its UART front end and both testbenches.

- **Run contract:** `G` / `Machine.start()` resets control state only; R, X and data memory keep their contents between runs. Programs must not read what neither they nor the host wrote; tests run with `Machine.scramble()` to check that.
- **Allocation:** the compiler uses R1..R23; R24..R31 and X4..X15 belong to the library. An interrupt handler's registers are disjoint from everything main uses.
- **Catalytic verification on machine code:** `analysis/isa_catalytic.py` checks the allocated handler (straight-line, writes only scratch and borrowed registers, linear ops on borrowed ones, net affine map over all registers is the identity). It runs inside `emit_isa`. The name-based `catalytic_verify.py` says nothing about allocated code.
- **Testbench guards:** the core testbench runs ~70 cases back to back on one core (state carried, as on the board); mutation tests in `test_isa_core.py` prove it can fail. Interrupt trigger points are retired-instruction counts (`irq_at`), matched in the testbench at instruction boundaries.

## Spec

`Cook_Mertz_Spec_v2.md` is the authoritative reference. Section 4 has the complete BNF grammar. The spec documents six v1 bugs fixed in v2; the most important are: `cyclo(n)` replaced by `roots(p)` (primitive roots, not cyclotomic polynomial), `IRRotate.theta: float` replaced by `IRRotate.j: int`, and `IRPoly` now carries an `IRField`.
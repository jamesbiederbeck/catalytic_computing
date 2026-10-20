"""Generate a Tang Nano 20K project for every example that lowers to hardware.

Usage: uv run python examples/fpga/build_projects.py <output-dir> [--debug] [slug ...]
       uv run python examples/fpga/build_projects.py <output-dir> --core
  e.g. <playground>/projects/cmtz

--core builds cmtz-core instead: the reusable processor (cmtz/isa/), one
bitstream for every program over F_17, loaded over the UART (see
examples/fpga/core_run.py). Always -noalu.

--debug builds the state-readback variant (cmtz-<name>-debug): same datapath
plus a dbg_state bus, and a UART front end with I / R / S commands for the
browser visualizer (visualizer/). Optional slugs limit what is built.

Each examples/NN_name.cmtz becomes <output-dir>/cmtz-name/ with the generated
datapath and its testbench, a generated UART front end and its testbench,
apio.ini, pins.cst, a diagram.svg of the datapath and a README. The literal embeds of a program become input
ports (`inputs="all"`), so the closed constant programs turn into real
pipelines. Examples that use matpow (12_cost_chain, fibonacci) are skipped;
fibonacci_pure stands in for fibonacci as `cmtz-fibonacci`.

An existing README in a project directory is left alone. cmtz-mul is separate
(examples/fpga/uart_mul/build.py). No board access.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for diagram.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # cmtz

from cmtz.compiler import compile_program
from cmtz.backends.verilog_backend import emit_verilog
from cmtz.backends.verilog_uart import emit_uart_wrapper
from cmtz.backends.isa_core import emit_core, emit_core_testbench, standard_cases
from cmtz.backends.isa_uart import (
    emit_core_uart, emit_core_uart_testbench, standard_uart_cases,
)
from cmtz.isa import CoreConfig
from diagram import draw_datapath

EXAMPLES = Path(__file__).resolve().parent.parent

PINS = """IO_LOC "clk" 4;
IO_PORT "clk" IO_TYPE=LVCMOS33 PULL_MODE=NONE;
IO_LOC "tx" 69;
IO_PORT "tx" IO_TYPE=LVCMOS33 PULL_MODE=UP;
IO_LOC "rx" 70;
IO_PORT "rx" IO_TYPE=LVCMOS33 PULL_MODE=UP;
""" + "".join(f'IO_LOC "leds[{i}]" {15 + i};\n' for i in range(6)) + "".join(
    f'IO_PORT "leds[{i}]" IO_TYPE=LVCMOS33;\n' for i in range(6))

# (source file, project slug); file order is the example order
SKIP = {"12_cost_chain", "fibonacci"}

# Default build flag is `yosys-extra-options = -noalu` (arithmetic in LUTs).
# A project gets yosys's default ALU mapping only if it is listed in
# ALU_VERIFIED, i.e. it was run on the board with that mapping and matched the
# reference. Measured on the board (GW2AR-18C, yosys 0.63, nextpnr-himbaechel,
# apicula): the two magsq projects gave wrong answers on the chip with the
# default mapping (67 and 92 of 96 vectors) although RTL and post-synthesis
# simulation passed; with -noalu all matched. The cause inside the toolchain
# is not isolated, so the list is empirical and a new example stays on -noalu
# until it has been checked both ways with examples/fpga/uart_check.py.
ALU_VERIFIED = {
    "cmtz-hello-field", "cmtz-rotate-fuse", "cmtz-conjugate-catalytic",
    "cmtz-add-superposition", "cmtz-interrupt", "cmtz-ntt-butterfly",
    "cmtz-interrupt-handler", "cmtz-fermat-little", "cmtz-diffie-hellman",
    "cmtz-belief-propagation", "cmtz-fibonacci",
    "cmtz-fibonacci-debug",
}

# Checked on the board over the UART (96 vectors, all matching, 128 for the
# debug builds) with the flags this script writes. A new example is not on
# this list until it has been run.
ON_BOARD = {
    "cmtz-hello-field", "cmtz-rotate-fuse", "cmtz-conjugate-catalytic",
    "cmtz-add-superposition", "cmtz-interrupt", "cmtz-phasor-basics",
    "cmtz-phasor-interference", "cmtz-ntt-butterfly", "cmtz-interrupt-handler",
    "cmtz-fermat-little", "cmtz-diffie-hellman", "cmtz-belief-propagation",
    "cmtz-fibonacci", "cmtz-fibonacci-debug", "cmtz-phasor-interference-debug",
    "cmtz-core",        # core_run.py: 2459 of 2459 checks, -noalu
}


def uses_noalu(project: str) -> bool:
    return project not in ALU_VERIFIED


def targets() -> list[tuple[Path, str]]:
    out = []
    for f in sorted(EXAMPLES.glob("*.cmtz")):
        if f.stem in SKIP:
            continue
        slug = "fibonacci" if f.stem == "fibonacci_pure" else \
            re.sub(r"^\d+_", "", f.stem).replace("_", "-")
        out.append((f, slug))
    return out


def blurb(source: str) -> str:
    """First prose line of the header comment."""
    for line in source.splitlines()[1:]:
        text = line.lstrip("-").strip()
        if text:
            return text
    return ""


def hexs(vals: list[int]) -> str:
    return " ".join(f"{v:02X}" for v in vals)


def readme(slug: str, src_name: str, source: str, bundle, wrap, mod: str) -> str:
    project = f"cmtz-{slug}"
    ins = [pt for pt in bundle.ports if pt.direction == "input"]
    outs = [pt for pt in bundle.ports if pt.direction == "output"]

    def rows(ports, kind):
        lines, k = [], 0
        for pt in ports:
            n = 2 if pt.is_complex else 1
            lanes = f"{k} and {k + 1} (re, im)" if n == 2 else f"{k}"
            lines.append(f"| `{pt.name}` | {pt.width} | {kind} byte {lanes} |")
            k += n
        return "\n".join(lines)

    il, ol = bundle.vectors[min(3, len(bundle.vectors) - 1)]
    p = bundle.field_p
    noalu = uses_noalu(project)
    noalu_note = (
        "## Build flag\n\n`apio.ini` sets `yosys-extra-options = -noalu`. With yosys's "
        "default ALU mapping this design gave wrong answers on the chip (most vectors "
        "mismatched) although RTL and post-synthesis simulation pass; with `-noalu` "
        "every vector matched. The cause inside the toolchain is not isolated. The "
        "price is up to twice the LUTs. A project only gets the default ALU mapping "
        "once it has been checked on the board with it.\n\n" if noalu else "")
    board_status = (
        "Not loaded on the board." if project not in ON_BOARD else
        "Checked on the board over the UART (96 vectors, all matching) with the "
        "flag above." if noalu else
        "Checked on the board over the UART (96 vectors, all matching).")
    return f"""# cmtz-{slug}

{blurb(source)}

The datapath is generated by the Verilog backend of the Cook-Mertz compiler from `{src_name}`:

```
{source.rstrip()}
```

The program's literal embeds are replaced by input ports, so the datapath computes the same graph for any values in 0 to {p - 1} instead of the constants in the source. It has {bundle.latency} pipeline stage{'s' * (bundle.latency != 1)}, one result per clock, and no `%` operators: reductions are unrolled conditional subtracts.

![Datapath of cmtz-{slug}: UART request bytes into {len(ins)} input port{'s' * (len(ins) != 1)}, the program's DAG over {bundle.latency} pipeline stage{'s' * (bundle.latency != 1)}, {len(outs)} output{'s' * (len(outs) != 1)} back out as UART reply bytes](diagram.svg)

## Ports

| Port | Bits | Carried as |
|---|---|---|
{rows(ins, "request")}
{rows(outs, "reply")}

## UART protocol

115200 8N1 on the on-board USB bridge. Send {wrap.request_bytes} byte{'s' * (wrap.request_bytes != 1)} (each 0 to {p - 1}), read {wrap.reply_bytes} byte{'s' * (wrap.reply_bytes != 1)} back. A complex port is two bytes, real part first. If any request byte is {p} or more, every reply byte is `0xFF`. If a request stalls partway for 250 ms, the bytes received so far are dropped. Bytes sent while a reply is going out are ignored, so read the whole reply before sending the next request.

Example: `{hexs(il)}` replies `{hexs(ol)}`.

LEDs (active low): `leds[4:0]` show the low bits of the first reply byte, `leds[5]` lights after an error reply.

## Files

| File | What |
|---|---|
| `{mod}_uart.v` | Top module `{mod}_uart`: UART receiver and transmitter and a control state machine around the datapath. Generated. |
| `{mod}.v` | The datapath. Generated. |
| `{mod}_tb.v` | Datapath testbench with {bundle.num_vectors} vector{'s' * (bundle.num_vectors != 1)}, expected values from the compiler's Python reference. Generated. |
| `{mod}_uart_tb.v` | Whole-design testbench: bit-bangs the UART protocol for every vector, then checks the out-of-range reply and the resync timeout. Generated. |
| `pins.cst` | `clk` pin 4, `tx` pin 69, `rx` pin 70, LEDs pins 15 to 20 |
| `apio.ini` | Board and top module |

Regenerate with `examples/fpga/build_projects.py` in the Cook-Mertz repo rather than editing the generated files.

{noalu_note}## Test

```
source ../../../env.sh
apio test        # prints PASS for both testbenches
```

## Status

Both testbenches pass in simulation and `apio build` succeeds. {board_status}
"""


def debug_readme(slug: str, src_name: str, bundle, wrap) -> str:
    project = f"cmtz-{slug}-debug"
    flag = ("`apio.ini` sets `yosys-extra-options = -noalu` (see `build_projects.py` "
            "for why).\n\n" if uses_noalu(project) else "")
    status = ("Checked on the board over the UART: info frame, hash, replies and "
              "every state byte matched." if project in ON_BOARD else
              "Not loaded on the board.")
    return f"""# cmtz-{slug}-debug

`{src_name}` as a pipelined datapath with state readback, for the browser
visualizer (`visualizer/` in the Cook-Mertz repo). The datapath is the same as
`cmtz-{slug}` plus a `dbg_state` bus carrying every input and registered node,
and the UART front end understands three commands (115200 8N1):

| Byte | Meaning |
|---|---|
| `I` (0x49) | 14-byte info frame: `CMTZ`, 1, request bytes ({wrap.request_bytes}), reply bytes ({wrap.reply_bytes}), state bytes, p ({bundle.field_p}), latency ({bundle.latency}), then the program hash (crc32, big endian) |
| `R` (0x52) | followed by {wrap.request_bytes} request byte{'s' * (wrap.request_bytes != 1)} (each 0 to {bundle.field_p - 1}); replies {wrap.reply_bytes} byte{'s' * (wrap.reply_bytes != 1)}. Any byte of {bundle.field_p} or more makes every reply byte `0xFF` |
| `S` (0x53) | one byte per lane of every input and registered node, in the order the visualizer shows; valid {bundle.latency} clock{'s' * (bundle.latency != 1)} after the last accepted request |

Other bytes are ignored, and a command that stalls for 250 ms is dropped.
![Datapath of {project}: UART request bytes into the program's DAG over {bundle.latency} pipeline stage{'s' * (bundle.latency != 1)} and reply bytes back out, with a state readback bus](diagram.svg)

Generated by `examples/fpga/build_projects.py --debug`; do not edit.

{flag}## Test

```
source ../../../env.sh
apio test
```

## Status

Both testbenches pass in simulation, including the info frame, the state
readback and the timeouts. {status}
"""


def build(outdir: Path, debug: bool = False,
          only: list[str] | None = None) -> None:
    for src_file, slug in targets():
        if only and slug not in only:
            continue
        source = src_file.read_text()
        result = compile_program(source, backend="python")
        assert result.ok, (src_file.name, result.errors)
        mod = "cmtz_" + slug.replace("-", "_") + ("_dbg" if debug else "")
        bundle = emit_verilog(result.ir_program, module_name=mod, inputs="all",
                              debug=debug)
        wrap = emit_uart_wrapper(bundle)
        d = outdir / f"cmtz-{slug}{'-debug' if debug else ''}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{mod}.v").write_text(bundle.verilog_source)
        (d / f"{mod}_tb.v").write_text(bundle.testbench_source)
        (d / f"{mod}_uart.v").write_text(wrap.source)
        (d / f"{mod}_uart_tb.v").write_text(wrap.testbench)
        (d / "apio.ini").write_text(
            f"[env:default]\nboard = sipeed-tang-nano-20k\n"
            f"top-module = {wrap.top_name}\n"
            + ("yosys-extra-options = -noalu\n"
               if uses_noalu(d.name) else ""))
        (d / "pins.cst").write_text(PINS)
        (d / "diagram.svg").write_text(draw_datapath(
            bundle, d.name, f"F_{bundle.field_p} datapath from {src_file.name}"))
        if not (d / "README.md").exists():
            (d / "README.md").write_text(
                debug_readme(slug, src_file.name, bundle, wrap) if debug else
                readme(slug, src_file.name, source, bundle, wrap, mod))
        print(f"{d.name}: {wrap.request_bytes} byte request, "
              f"{wrap.reply_bytes} byte reply, latency {bundle.latency}")


CORE = CoreConfig(17, 19)


def core_readme(core: CoreConfig, project: str) -> str:
    status = ("Checked on the board with `examples/fpga/core_run.py` (2459 of 2459 checks,"
              " including the roots-of-unity projection programs; see"
              " `docs/cmtz-projection.md` of this repo)."
              if project in ON_BOARD else "Not loaded on the board.")
    return f"""# {project}

A reusable Cook-Mertz processor for F_{core.p} (q = {core.q}). Unlike the other
`cmtz-*` projects, where every program is its own circuit, this bitstream runs
any program the Cook-Mertz compiler emits for this field, loaded over the UART
in milliseconds: compiled `.cmtz` programs, `matpow` with the matrix written
to data memory at run time, and hand-written assembly. Interrupt handlers
(`interrupt {{ }}`) run on the live register file; the compiler proves on the
allocated machine code that they restore the registers they borrow.

The ISA, assembler, simulator and compiler backend are in `cmtz/isa/` and
`cmtz/backends/isa_backend.py` of the Cook-Mertz repo; the design notes are
in `docs/cmtz-processor.md` of this repo.

| | |
|---|---|
| Program memory | {core.pmem_size} x 32-bit instructions |
| Data memory | {core.dmem_size} x {core.rw} bits |
| Registers | R0..R31 field ({core.rw} bits, R0 = 0), X0..X15 index (16 bits, X0 = 0) |
| Return stack | 8 |
| Interrupts | one handler, no nesting, triggered over the UART |

## UART protocol

115200 8N1. One command byte, then arguments (big endian); the full
description is the docstring of `cmtz/isa/host.py`.

| Byte | Meaning |
|---|---|
| `I` | 21-byte info frame: `CMTC`, ISA version, p, q, memory sizes, core hash, loaded program length and crc32 |
| `P` | load a program: n, entry, handler (`FFFF` = none), n x 4 bytes; replies `00` + crc32 of the words |
| `W` | write a field register |
| `D` / `M` | write / read data memory |
| `S` | read R0..R31 and X0..X15 (64 bytes) |
| `G` | run; replies `00` at once |
| `Q` | 16-byte status: state, error, retired instructions, interrupt count, last dispatch |
| `T` | trigger the interrupt (works while running) |
| `X` | abort a run |

`0xFF` marks a refused command (every reply byte). A run resets the control
state only; registers and data memory keep their contents between runs.

LEDs (active low): 0 running, 1 halted, 2 error, 3 toggles per interrupt,
4 program loaded, 5 last reply was an error.

## Files

| File | What |
|---|---|
| `cmtz_core.v` | The processor. Generated. |
| `cmtz_core_uart.v` | Top module `cmtz_core_uart`: UART receiver, transmitter and the command state machine. Generated. |
| `cmtz_core_tb.v` | Core testbench: every real-field example, matpow, interrupts, an exhaustive sweep of every field instruction and every error code, run back to back on one core and checked against the simulator. Generated. |
| `cmtz_core_uart_tb.v` | Whole-design testbench over the UART pins, including an interrupt that lands mid-loop and the refusals. Generated. |
| `pins.cst` | `clk` pin 4, `tx` pin 69, `rx` pin 70, LEDs pins 15 to 20 |
| `apio.ini` | Board, top module, `-noalu` |

Regenerate with `examples/fpga/build_projects.py <dir> --core` in the
Cook-Mertz repo rather than editing the generated files.

## Build flag

`apio.ini` sets `yosys-extra-options = -noalu`, the default for every cmtz
project until it has been checked on the board both ways (see
`build_projects.py`).

## Test

```
source ../../../env.sh
apio test        # prints PASS for both testbenches
```

## Status

Both testbenches pass in simulation and `apio build` succeeds. {status}
"""


def build_core(outdir: Path, core: CoreConfig = CORE) -> Path:
    project = "cmtz-core"
    d = outdir / project
    d.mkdir(parents=True, exist_ok=True)
    tb = emit_core_testbench(core, standard_cases(core, EXAMPLES))
    wrap = emit_core_uart(core)
    (d / "cmtz_core.v").write_text(emit_core(core))
    (d / "cmtz_core_tb.v").write_text(tb.source)
    (d / "cmtz_core_uart.v").write_text(wrap.source)
    (d / "cmtz_core_uart_tb.v").write_text(
        emit_core_uart_testbench(core, standard_uart_cases(core, EXAMPLES)))
    (d / "apio.ini").write_text(
        "[env:default]\nboard = sipeed-tang-nano-20k\n"
        f"top-module = {wrap.top_name}\nyosys-extra-options = -noalu\n")
    (d / "pins.cst").write_text(PINS)
    if not (d / "README.md").exists():
        (d / "README.md").write_text(core_readme(core, project))
    print(f"{project}: {tb.cases} core cases")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--core", action="store_true")
    ap.add_argument("slugs", nargs="*")
    args = ap.parse_intermixed_args()
    if args.core:
        build_core(args.outdir)
    else:
        build(args.outdir, args.debug, args.slugs)

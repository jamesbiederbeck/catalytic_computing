"""Generate the cmtz-mul board project: datapath from the compiler + UART wrapper.

Usage: uv run python examples/fpga/uart_mul/build.py <output-dir>
  e.g. <playground>/projects/cmtz/cmtz-mul

Writes the generated datapath (cmtz_mul.v) and its self-checking testbench
(cmtz_mul_tb.v) and a diagram.svg of the datapath, copies the hand-written UART
wrapper and its testbench, and writes apio.ini, pins.cst and README.md (only if absent, so a README edited
in the destination, e.g. with hardware results, is kept). No board access.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # for diagram.py

from cmtz.compiler import compile_program
from cmtz.backends.verilog_backend import emit_verilog
from diagram import draw_datapath

SOURCE = """
field(17, 293);
embed(0, 1);
embed(1, 5);
mul(embed_0, embed_1) as m;
measure(m, mod_p) as product;
"""

APIO_INI = """[env:default]
board = sipeed-tang-nano-20k
top-module = cmtz_mul_uart
"""

PINS = """IO_LOC "clk" 4;
IO_PORT "clk" IO_TYPE=LVCMOS33 PULL_MODE=NONE;
IO_LOC "tx" 69;
IO_PORT "tx" IO_TYPE=LVCMOS33 PULL_MODE=UP;
IO_LOC "rx" 70;
IO_PORT "rx" IO_TYPE=LVCMOS33 PULL_MODE=UP;
""" + "".join(f'IO_LOC "leds[{i}]" {15 + i};\n' for i in range(6)) + "".join(
    f'IO_PORT "leds[{i}]" IO_TYPE=LVCMOS33;\n' for i in range(6))


def main(outdir: Path) -> None:
    here = Path(__file__).parent
    result = compile_program(SOURCE, backend="python")
    assert result.ok, result.errors
    bundle = emit_verilog(result.ir_program, module_name="cmtz_mul")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cmtz_mul.v").write_text(bundle.verilog_source)
    (outdir / "cmtz_mul_tb.v").write_text(bundle.testbench_source)
    for name in ("cmtz_mul_uart.v", "cmtz_mul_uart_tb.v"):
        shutil.copy(here / name, outdir / name)
    if not (outdir / "README.md").exists():
        shutil.copy(here / "README.md", outdir / "README.md")
    (outdir / "diagram.svg").write_text(draw_datapath(
        bundle, "cmtz-mul", f"F_{bundle.field_p} datapath, hand-written UART wrapper"))
    (outdir / "apio.ini").write_text(APIO_INI)
    (outdir / "pins.cst").write_text(PINS)
    print(f"wrote {outdir} (datapath latency {bundle.latency} clocks)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

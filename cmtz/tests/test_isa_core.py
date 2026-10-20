"""cmtz-core RTL and UART front end (simulated with iverilog against sim.py)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from cmtz.backends.isa_core import (
    CoreCase, emit_core, emit_core_testbench, standard_cases,
)
from cmtz.backends.isa_uart import (
    emit_core_uart, emit_core_uart_testbench, standard_uart_cases,
)
from cmtz.isa import CoreConfig, assemble
from cmtz.isa.encoding import OPS
from cmtz.isa.image import ProgramImage

IVERILOG = shutil.which("iverilog") and shutil.which("vvp")
EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"
CORE = CoreConfig(17, 19)


def _sim(tmp_path, files: dict[str, str]) -> str:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    subprocess.run(["iverilog", "-g2005", "-o", "s.out", *files],
                   cwd=tmp_path, check=True, capture_output=True)
    return subprocess.run(["vvp", "s.out"], cwd=tmp_path, check=True,
                          capture_output=True, text=True).stdout


def _verdict(out: str) -> str:
    lines = [l for l in out.splitlines() if l.startswith(("PASS", "FAIL"))]
    return lines[-1] if lines else "no verdict"


class TestEmission:
    def test_opcodes_come_from_encoding(self):
        v = emit_core(CORE)
        for name, spec in OPS.items():
            assert f"localparam OP_{name} = 6'd{spec.code};" in v
            assert f"OP_{name}:" in v or name in ("NOP",)

    def test_no_modulo_operator(self):
        assert "%" not in emit_core(CORE).split("module", 1)[1]

    def test_uart_needs_small_q(self):
        with pytest.raises(ValueError, match="q <= 254"):
            emit_core_uart(CoreConfig(251, 257))

    def test_testbench_rejects_unfinished_case(self):
        a = assemble(".handler isr\nWAITI\nHALT\nisr: RETI")
        img = ProgramImage(17, 19, a.words, a.entry, a.handler)
        with pytest.raises(ValueError, match="does not finish"):
            emit_core_testbench(CORE, [CoreCase(img)])


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestCoreSimulation:
    def test_standard_cases_pass(self, tmp_path):
        tb = emit_core_testbench(CORE, standard_cases(CORE, EXAMPLES))
        out = _sim(tmp_path, {"cmtz_core.v": emit_core(CORE),
                              "cmtz_core_tb.v": tb.source})
        assert _verdict(out).startswith("PASS"), out[-2000:]

    def test_wrong_expectations_fail(self, tmp_path):
        tb = emit_core_testbench(CORE, standard_cases(CORE, EXAMPLES),
                                 break_alu=True)
        out = _sim(tmp_path, {"cmtz_core.v": emit_core(CORE),
                              "cmtz_core_tb.v": tb.source})
        assert _verdict(out).startswith("FAIL")

    @pytest.mark.parametrize("old,new", [
        ("OP_SUB:    begin wr(sub_r)", "OP_SUB:    begin wr(add_r)"),
        ("OP_WAITI:  if (handled)", "OP_WAITI:  if (1'b1)"),
        ("if (fa != 5'd0) R[fa] <= v;", "R[fa] <= v;"),
        ("if (irq) pending <= 1'b1;", ""),
    ])
    def test_broken_core_fails(self, tmp_path, old, new):
        v = emit_core(CORE)
        assert old in v
        tb = emit_core_testbench(CORE, standard_cases(CORE, EXAMPLES))
        out = _sim(tmp_path, {"cmtz_core.v": v.replace(old, new),
                              "cmtz_core_tb.v": tb.source})
        assert _verdict(out).startswith("FAIL")


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestUartSimulation:
    def _files(self, mutate=None):
        top = emit_core_uart(CORE).source
        return {"cmtz_core.v": emit_core(CORE),
                "cmtz_core_uart.v": top if mutate is None else mutate(top),
                "cmtz_core_uart_tb.v": emit_core_uart_testbench(
                    CORE, standard_uart_cases(CORE, EXAMPLES))}

    def test_protocol_passes(self, tmp_path):
        out = _sim(tmp_path, self._files())
        assert _verdict(out).startswith("PASS"), out[-2000:]

    @pytest.mark.parametrize("old,new", [
        ("crc <= crc8(crc, rx_data);", "crc <= crc;"),
        ("x_addr <= idx[4:1];", "x_addr <= idx[3:0];"),
        ("\"T\": begin irq <= 1'b1;", "\"T\": begin irq <= 1'b0;"),
    ])
    def test_broken_front_end_fails(self, tmp_path, old, new):
        def mutate(src):
            assert old in src
            return src.replace(old, new)
        out = _sim(tmp_path, self._files(mutate))
        assert _verdict(out).startswith("FAIL")

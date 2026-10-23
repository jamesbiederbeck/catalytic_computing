"""cmtz-core RTL and UART front end (simulated with iverilog against sim.py)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from cmtz.backends.isa_core import (
    CoreCase, emit_core, emit_core_testbench, standard_cases,
)
from cmtz.backends.isa_uart import (
    emit_core_step_testbench, emit_core_uart, emit_core_uart_testbench,
    standard_step_cases, standard_uart_cases,
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


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestProjectionSimulation:
    """The projection programs on the RTL, state and cycles checked against
    the simulator (which the oracle tests already tie to the math)."""

    def test_projection_cases_pass(self, tmp_path):
        from cmtz.isa.projection import rtl_cases
        tb = emit_core_testbench(CORE, rtl_cases())
        out = _sim(tmp_path, {"cmtz_core.v": emit_core(CORE),
                              "cmtz_core_tb.v": tb.source})
        assert _verdict(out).startswith("PASS"), out[-2000:]

    def test_wrong_cycle_count_fails(self, tmp_path):
        from cmtz.isa.projection import rtl_cases
        cases = rtl_cases()[:1]
        tb = emit_core_testbench(CORE, cases).source
        n = tb.split("chk_cycles(")[2].split(")")[0]
        out = _sim(tmp_path, {"cmtz_core.v": emit_core(CORE),
                              "cmtz_core_tb.v": tb.replace(
                                  f"chk_cycles({n});", f"chk_cycles({int(n) + 1});")})
        assert _verdict(out).startswith("FAIL")


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestCycleModel:
    """`Machine.cycles` against the RTL on the paths the projection
    programs do not take: loads, a trigger with no handler, faults."""

    def test_cycles_match(self, tmp_path):
        from cmtz.backends.isa_backend import emit_isa
        from cmtz.compiler import compile_program

        r = compile_program((EXAMPLES / "fibonacci.cmtz").read_text(), backend="python")
        fib = emit_isa(r.ir_program, CORE, matrix_dims={"fib_matrix": 2})

        def asm(src, **kw):
            a = assemble(src)
            return ProgramImage(17, 19, a.words, a.entry, a.handler, **kw)

        cases = [
            CoreCase(fib.image, "fibonacci matpow (LD)",
                     # no zeros: the testbench writes only words that differ from the
                     # simulator's zero-filled memory, and RTL memory starts as x
                     matrices={"fib_matrix": [[1, 1], [1, 2]]}, check_cycles=True),
            CoreCase(asm("LIX X1, 0\nLD R1, [X1]\nLD R2, [X1+1]\nHALT"),
                     "two loads", check_cycles=True),
            CoreCase(asm("NOP\nNOP\nNOP\nNOP\nHALT"), "trigger, no handler",
                     irq_at=(2,), check_cycles=True),
            CoreCase(asm(f"LIX X1, {CORE.dmem_size}\nLD R1, [X1]\nHALT"),
                     "LD address fault", check_cycles=True),
            CoreCase(asm("NOP"), "pc fault", check_cycles=True),
            CoreCase(asm("f: CALL f"), "stack overflow", check_cycles=True),
        ]
        tb = emit_core_testbench(CORE, cases)
        assert tb.expected[0].cycles > 2 * tb.expected[0].retired   # has loads
        out = _sim(tmp_path, {"cmtz_core.v": emit_core(CORE),
                              "cmtz_core_tb.v": tb.source})
        assert _verdict(out).startswith("PASS"), out[-2000:]


class TestStepBuildIsOptIn:
    """The debugger is a flag; without it the generated Verilog has none of it."""

    def test_default_core_has_no_pause(self):
        src = emit_core(CORE)
        for word in ("S_PAUSE", "pause_req", "step_req", "resume_req", "paused"):
            assert word not in src

    def test_default_front_end_has_no_step_commands(self):
        src = emit_core_uart(CORE).source
        for word in ("pause_req", "step_req", "resume_req", "status_b", "blink"):
            assert word not in src
        assert '"H"' not in src and '"N"' not in src

    def test_step_core_has_them(self):
        assert "S_PAUSE" in emit_core(CORE, step=True)
        assert '"H", "N"' in emit_core_uart(CORE, step=True).source

    def test_lane_cannot_step(self):
        with pytest.raises(ValueError):
            emit_core(CORE, lane=True, step=True)


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestStepBuildSimulation:
    """cmtz-core with the debugger: every `N` is compared with Machine.step()."""

    def _files(self, mutate_core=None, tb="step"):
        core = emit_core(CORE, step=True)
        files = {"cmtz_core.v": core if mutate_core is None else mutate_core(core),
                 "cmtz_core_uart.v": emit_core_uart(CORE, step=True).source}
        if tb == "step":
            files["cmtz_core_step_tb.v"] = emit_core_step_testbench(
                CORE, standard_step_cases(CORE, EXAMPLES))
        else:
            files["cmtz_core_uart_tb.v"] = emit_core_uart_testbench(
                CORE, standard_uart_cases(CORE, EXAMPLES))
        return files

    def test_stepping_matches_the_simulator(self, tmp_path):
        out = _sim(tmp_path, self._files())
        assert _verdict(out).startswith("PASS"), out[-2000:]

    def test_every_old_command_still_behaves(self, tmp_path):
        # the unchanged UART testbench, on the step build
        out = _sim(tmp_path, self._files(tb="uart"))
        assert _verdict(out).startswith("PASS"), out[-2000:]

    @pytest.mark.parametrize("old,new", [
        # ordinary instructions never complete a step
        ("state <= S_FETCH; step_done <= 1'b1;", "state <= S_FETCH;"),
        # a pause asked for while running is dropped
        ("pause_pend <= 1'b1;", "pause_pend <= 1'b0;"),
        # resume restarts from the entry instead of continuing
        ("stepping <= 1'b0; state <= S_FETCH;",
         "stepping <= 1'b0; pc <= cfg_entry; state <= S_FETCH;"),
    ])
    def test_broken_step_core_fails(self, tmp_path, old, new):
        def mutate(src):
            assert old in src
            return src.replace(old, new)
        out = _sim(tmp_path, self._files(mutate))
        assert _verdict(out).startswith("FAIL")

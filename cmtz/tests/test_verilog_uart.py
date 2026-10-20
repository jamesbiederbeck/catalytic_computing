"""Tests for the UART wrapper generator (simulated with iverilog)."""

import shutil
import subprocess

import pytest

from cmtz.compiler import compile_program
from cmtz.backends.verilog_backend import emit_verilog
from cmtz.backends.verilog_uart import emit_uart_wrapper

IVERILOG = shutil.which("iverilog") and shutil.which("vvp")

REAL = "field(17, 293); embed(0,1); embed(1,5); mul(embed_0, embed_1) as m; " \
       "measure(m, mod_p) as product;"
CPLX = ("cfield(7); cembed(0, 1); cembed(2, 0); add(cembed_1, cembed_2) as s; "
        "conj(s) as c; magsq(s) as power; measure(power, mod_p) as norm;")


def _wrap(src, **kw):
    r = compile_program(src, backend='python')
    assert r.ok, r.errors
    b = emit_verilog(r.ir_program, module_name="dut", **kw)
    return b, emit_uart_wrapper(b, top_name="top")


def _sim(b, w, tmp_path, mutate=None) -> str:
    top = w.source if mutate is None else mutate(w.source)
    for name, text in (("dut.v", b.verilog_source), ("top.v", top),
                       ("top_tb.v", w.testbench)):
        (tmp_path / name).write_text(text)
    subprocess.run(["iverilog", "-g2005", "-o", "s.out", "dut.v", "top.v",
                    "top_tb.v"], cwd=tmp_path, check=True, capture_output=True)
    return subprocess.run(["vvp", "s.out"], cwd=tmp_path, check=True,
                          capture_output=True, text=True).stdout


class TestWrapperEmission:
    def test_lane_counts(self):
        _, w = _wrap(REAL)
        assert (w.request_bytes, w.reply_bytes) == (2, 1)
        _, w = _wrap(CPLX, inputs="all")
        assert (w.request_bytes, w.reply_bytes) == (4, 1)   # two complex inputs

    def test_needs_inputs_and_outputs(self):
        r = compile_program("field(17); embed(0, 3); "
                            "measure(embed_0, mod_p) as r;", backend='python')
        b = emit_verilog(r.ir_program)          # no IRMul: no input ports
        with pytest.raises(ValueError, match="at least one input"):
            emit_uart_wrapper(b)

    def test_large_prime_rejected(self):
        r = compile_program("field(257, 263); embed(0, 1); embed(1, 2); "
                            "mul(embed_0, embed_1) as m; measure(m, mod_p) as o;",
                            backend='python')
        with pytest.raises(ValueError, match="one byte per lane"):
            emit_uart_wrapper(emit_verilog(r.ir_program))


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestWrapperSimulation:
    @pytest.mark.parametrize("src,kw", [(REAL, {}), (CPLX, {"inputs": "all"})],
                             ids=["real", "complex"])
    def test_whole_design_passes(self, src, kw, tmp_path):
        b, w = _wrap(src, **kw)
        assert "PASS" in _sim(b, w, tmp_path)

    def test_bench_fails_without_resync(self, tmp_path):
        b, w = _wrap(REAL)
        out = _sim(b, w, tmp_path, mutate=lambda s: s.replace(
            "if (idle == TIMEOUT_CLKS - 1)", "if (1'b0)"))
        assert "PASS" not in out

    def test_bench_fails_without_range_check(self, tmp_path):
        b, w = _wrap(REAL)
        out = _sim(b, w, tmp_path, mutate=lambda s: s.replace(
            "if (rx_data >= 8'd17) bad <= 1'b1;", ""))
        assert "FAIL" in out


# ── debug protocol: info / run / state ───────────────────────────────────────

from pathlib import Path

from cmtz.backends.verilog_uart import lane_table, info_frame

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _dbg(src, **kw):
    kw.setdefault("inputs", "all")
    r = compile_program(src, backend='python')
    assert r.ok, r.errors
    b = emit_verilog(r.ir_program, module_name="dut", debug=True, **kw)
    return b, emit_uart_wrapper(b, top_name="top")


class TestDebugEmission:
    def test_datapath_has_state_bus(self):
        b, _ = _dbg(REAL, inputs=None)
        assert b.debug and b.dbg_width > 0
        assert "output [" in b.verilog_source and "dbg_state" in b.verilog_source

    def test_plain_bundle_has_no_state_bus(self):
        r = compile_program(REAL, backend='python')
        b = emit_verilog(r.ir_program)
        assert not b.debug and "dbg_state" not in b.verilog_source
        assert all(n.dbg_offset is None for n in b.nodes)

    def test_manifest_kinds_and_depths(self):
        b, _ = _dbg(REAL, inputs=None)
        kinds = {n.name: (n.kind, n.depth) for n in b.nodes}
        assert kinds["embed_0"] == ("input", 0)
        assert kinds["m"] == ("reg", 1)
        assert kinds["product"][0] == "alias"

    def test_hash_depends_on_structure(self):
        a = _dbg(REAL)[0]
        assert a.program_hash == _dbg(REAL)[0].program_hash
        # embed constants become ports here, so they do not change the hardware
        assert a.program_hash == _dbg(REAL.replace("embed(1,5)", "embed(1,6)"))[0].program_hash
        # a different operation does
        assert a.program_hash != _dbg(REAL.replace("mul(", "add("))[0].program_hash

    def test_lane_table(self):
        b, _ = _dbg(CPLX)
        t = lane_table(b)
        assert [(x["node"], x["lane"]) for x in t["inputs"]] == [
            ("cembed_1", "re"), ("cembed_1", "im"),
            ("cembed_2", "re"), ("cembed_2", "im")]
        assert t["state"][0] == {"node": "cembed_1", "lane": "re"}

    def test_info_frame_layout(self):
        b, w = _dbg(REAL, inputs=None)
        f = info_frame(b, w.request_bytes, w.reply_bytes, len(lane_table(b)["state"]))
        assert bytes(f[:4]) == b"CMTZ" and len(f) == 14
        assert int.from_bytes(bytes(f[10:]), "big") == b.program_hash


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestDebugSimulation:
    @pytest.mark.parametrize("src,kw", [
        (REAL, {"inputs": None}),
        (CPLX, {}),
        ((EXAMPLES / "fibonacci_pure.cmtz").read_text(), {}),
        ((EXAMPLES / "05_interrupt.cmtz").read_text(), {}),
    ], ids=["mul", "complex", "fibonacci-10-stage", "interrupt"])
    def test_whole_design_passes(self, src, kw, tmp_path):
        b, w = _dbg(src, **kw)
        assert "PASS" in _sim(b, w, tmp_path)

    def test_bench_fails_on_swapped_state_bytes(self, tmp_path):
        b, w = _dbg(REAL, inputs=None)
        out = _sim(b, w, tmp_path, mutate=lambda s: s.replace(
            "8'd0: tx_byte = dbgb_0;", "8'd0: tx_byte = dbgb_1;"))
        assert "FAIL" in out

    def test_bench_fails_on_wrong_info(self, tmp_path):
        b, w = _dbg(REAL, inputs=None)
        out = _sim(b, w, tmp_path, mutate=lambda s: s.replace(
            "8'd4: tx_byte = 8'd1;", "8'd4: tx_byte = 8'd9;"))
        assert "FAIL" in out

    def test_bench_fails_without_command_timeout(self, tmp_path):
        b, w = _dbg(REAL, inputs=None)
        out = _sim(b, w, tmp_path, mutate=lambda s: s.replace(
            "if (idle == TIMEOUT_CLKS - 1)", "if (1'b0)"))
        assert "PASS" not in out

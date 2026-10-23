"""RTL simulation of the generated ML-KEM NTT core and UART top (iverilog)."""
import re
import shutil
import subprocess

import pytest

from cmtz.accel.mlkem_ntt import arith, gowin_sim, rtlgen, tb_core, tb_uart, uart_top
from cmtz.accel.mlkem_ntt.config import NTTConfig

IVERILOG = shutil.which("iverilog") and shutil.which("vvp")
pytestmark = pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")


def sim(tmp_path, files, top_files):
    for n, t in files.items():
        (tmp_path / n).write_text(t)
    subprocess.run(["iverilog", "-g2012", "-o", "s.out", *top_files], cwd=tmp_path,
                   check=True, capture_output=True)
    return subprocess.run(["vvp", "s.out"], cwd=tmp_path, check=True, capture_output=True,
                          text=True).stdout


class TestCoreSimulation:
    @pytest.mark.parametrize("lanes", [1, 2, 4])
    @pytest.mark.parametrize("reduction", ["montgomery", "barrett"])
    def test_core_passes(self, lanes, reduction, tmp_path):
        cfg = NTTConfig(lanes=lanes, reduction=reduction)
        out = sim(tmp_path, {"b.v": arith.emit_butterfly(reduction),
                             "c.v": rtlgen.emit_core(cfg)[0],
                             "t.v": tb_core.emit_core_tb(cfg)}, ["b.v", "c.v", "t.v"])
        assert "PASS mlkem_ntt_core" in out, out[-800:]

    def test_core_8_lanes(self, tmp_path):
        cfg = NTTConfig(lanes=8)
        out = sim(tmp_path, {"b.v": arith.emit_butterfly("montgomery"),
                             "c.v": rtlgen.emit_core(cfg)[0],
                             "t.v": tb_core.emit_core_tb(cfg)}, ["b.v", "c.v", "t.v"])
        assert "PASS mlkem_ntt_core" in out, out[-800:]

    def test_bench_detects_corrupted_rom(self, tmp_path):
        cfg = NTTConfig(lanes=4)
        core = rtlgen.emit_core(cfg)[0]
        m = re.search(r"(rom\[100\] = 188'h)([0-9a-f]+)", core)
        flipped = format(int(m.group(2), 16) ^ (1 << 150), "x")     # a lane-valid / select bit
        core = core.replace(m.group(0), m.group(1) + flipped)
        out = sim(tmp_path, {"b.v": arith.emit_butterfly("montgomery"), "c.v": core,
                             "t.v": tb_core.emit_core_tb(cfg)}, ["b.v", "c.v", "t.v"])
        assert "FAIL" in out and "PASS" not in out

    def test_bench_detects_early_write_register(self, tmp_path):
        cfg = NTTConfig(lanes=4)
        core = rtlgen.emit_core(cfg)[0].replace("wvb0 ? wdr0 :", "wvb0 ? wdsel0 :")
        out = sim(tmp_path, {"b.v": arith.emit_butterfly("montgomery"), "c.v": core,
                             "t.v": tb_core.emit_core_tb(cfg)}, ["b.v", "c.v", "t.v"])
        assert "FAIL" in out and "PASS" not in out

    @pytest.mark.parametrize("lanes", [1, 4])
    def test_sdp_primitive_banks(self, lanes, tmp_path):
        """Hand-instantiated SDPX9B banks (READ_MODE=1) behave like the inferred ones."""
        cfg = NTTConfig(lanes=lanes, bank_prim="sdp")
        out = sim(tmp_path, {"b.v": arith.emit_butterfly("montgomery"),
                             "c.v": rtlgen.emit_core(cfg)[0], "m.v": gowin_sim.MODEL,
                             "t.v": tb_core.emit_core_tb(cfg)}, ["m.v", "b.v", "c.v", "t.v"])
        assert "PASS mlkem_ntt_core" in out, out[-800:]

    def test_fast_clock_variants(self, tmp_path):
        """xreg/wreg off: shorter pipeline, same results, cycle model still exact."""
        cfg = NTTConfig(lanes=4, xreg=0, wreg=0)
        out = sim(tmp_path, {"b.v": arith.emit_butterfly("montgomery"),
                             "c.v": rtlgen.emit_core(cfg)[0],
                             "t.v": tb_core.emit_core_tb(cfg)}, ["b.v", "c.v", "t.v"])
        assert "PASS mlkem_ntt_core" in out, out[-800:]


class TestUartSimulation:
    @pytest.mark.parametrize("lanes", [1, 4])
    def test_uart_passes(self, lanes, tmp_path):
        cfg = NTTConfig(lanes=lanes)
        out = sim(tmp_path, {"b.v": arith.emit_butterfly("montgomery"),
                             "c.v": rtlgen.emit_core(cfg)[0], "u.v": uart_top.emit_top(cfg),
                             "t.v": tb_uart.emit_uart_tb(cfg)}, ["b.v", "c.v", "u.v", "t.v"])
        assert "PASS mlkem_ntt_uart" in out, out[-800:]

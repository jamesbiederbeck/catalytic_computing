"""Multilane cmtz-core RTL (RFC-003 A1/A2): generator, model agreement, and
iverilog runs of the direct and UART testbenches."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from cmtz.backends.isa_core import emit_core
from cmtz.backends.isa_mlane import (
    emit_mlane, emit_mlane_testbench, emit_mlane_uart,
    emit_mlane_uart_testbench, standard_scenes, uart_scenes,
)
from cmtz.isa import CoreConfig, Machine
from cmtz.isa import mlane_host as H

CORE = CoreConfig(17, 19)
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
IVERILOG = shutil.which("iverilog") and shutil.which("vvp")

# sha256 of the cmtz-core Verilog the board regression ran on (cmtz-core in
# the playground); the multilane work must not change it
CORE_V_SHA256 = "cc4518020b65c1242d6a96ee1e3741d0a570de35e31befbde64b951fcc94417b"


def sim(tmp_path, files: dict[str, str]) -> str:
    for name, src in files.items():
        (tmp_path / name).write_text(src)
    subprocess.run(["iverilog", "-g2005", "-o", "s.out", *files], cwd=tmp_path,
                   check=True, capture_output=True)
    return subprocess.run(["vvp", "s.out"], cwd=tmp_path, check=True,
                          capture_output=True, text=True, timeout=1800).stdout


class TestGenerator:
    def test_single_core_verilog_unchanged(self):
        src = emit_core(CORE)
        assert hashlib.sha256(src.encode()).hexdigest() == CORE_V_SHA256

    def test_independent_lanes_are_unmodified_cores(self):
        ml = emit_mlane(CORE, 2, lease=False)
        assert ml.lane_module == "cmtz_core"
        assert ml.lane_source == emit_core(CORE)
        assert "RF [" not in ml.source

    def test_lane_variant_keeps_the_datapath(self):
        lane = emit_core(CORE, "cmtz_lane", lane=True)
        core = emit_core(CORE)
        # everything outside the register file is the same text
        for line in core.splitlines():
            if "R[" in line or "R [" in line or "r_we" in line or "r_addr" in line \
                    or "r_wdata" in line or "r_rdata" in line \
                    or "module cmtz_core" in line or "S_EXEC: case" in line:
                continue
            assert line in lane, line

    def test_host_frames(self):
        info = H.info_frame(CORE, 2, True, 1, None)
        assert len(info) == H.INFO_LEN and info[:4] == b"CMTM"
        assert info[H.INFO_FIXED] == 1
        st = H.LaneStatus(2, 0, 307, 0, 0, 0, 22, 100, 714, 5)
        assert H.LaneStatus.parse(st.frame()) == st and st.cycles == 614
        assert H.mlane_hash(CORE, 2, True) != H.mlane_hash(CORE, 2, False)


class TestModelAgainstScalar:
    """The scenes the RTL is checked against: with no lease, every lane's
    expected result is what a scalar Machine computes on its own."""

    @pytest.mark.parametrize("lanes", [2, 4])
    def test_independent_scenes(self, lanes):
        from cmtz.backends.isa_mlane import run_scene, scene_model
        mc = scene_model(CORE, lanes)
        for sc in standard_scenes(CORE, lanes, False, EXAMPLES):
            R0 = list(mc.P)
            X0 = [list(ln.X) for ln in mc.lanes]
            D0 = [list(ln.D) for ln in mc.lanes]
            r = run_scene(mc, sc, mc.t + 1)
            for run in sc.runs:
                m = Machine(CORE, run.image)
                m.R = R0[run.lane * 32:(run.lane + 1) * 32]
                m.X, m.D = X0[run.lane], D0[run.lane]
                m.set_inputs({k: v for k, v in run.inputs.items()
                              if k in run.image.inputs})
                m.run(irq_at=run.irq_at, max_steps=10_000_000)
                res = r["results"][run.lane]
                assert (res.status, res.error, res.retired, res.cycles,
                        res.stalls) == (m.status, m.error, m.retired,
                                        m.cycles, 0), sc.name
                assert mc.P[run.lane * 32:(run.lane + 1) * 32] == m.R
                assert mc.lanes[run.lane].X == m.X
                assert mc.lanes[run.lane].D == m.D


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestMlaneSimulation:
    @pytest.mark.parametrize("lanes", [2, 4])
    def test_independent_lanes(self, tmp_path, lanes):
        ml = emit_mlane(CORE, lanes, False)
        tb = emit_mlane_testbench(CORE, lanes, False,
                                  standard_scenes(CORE, lanes, False, EXAMPLES))
        out = sim(tmp_path, {"core.v": ml.lane_source, "ml.v": ml.source,
                             "tb.v": tb.source})
        assert f"PASS: {tb.scenes} scenes" in out, out[-2000:]

    def test_independent_uart(self, tmp_path):
        ml = emit_mlane(CORE, 2, False)
        sc = uart_scenes(CORE, 2, False)
        out = sim(tmp_path, {
            "core.v": ml.lane_source, "ml.v": ml.source,
            "top.v": emit_mlane_uart(CORE, 2, False),
            "tb.v": emit_mlane_uart_testbench(CORE, 2, False, sc)})
        assert f"PASS: {len(sc)} scenes over the UART" in out, out[-2000:]

    @pytest.mark.parametrize("lanes", [2, 4])
    def test_leases(self, tmp_path, lanes):
        ml = emit_mlane(CORE, lanes, True)
        tb = emit_mlane_testbench(CORE, lanes, True,
                                  standard_scenes(CORE, lanes, True, EXAMPLES))
        out = sim(tmp_path, {"lane.v": ml.lane_source, "ml.v": ml.source,
                             "tb.v": tb.source})
        assert f"PASS: {tb.scenes} scenes" in out, out[-2000:]
        stalls = [r.stalls for sc in tb.results for r in sc["results"].values()]
        assert max(stalls) > 30000          # the tree scene stalls the owner
        codes = [c for sc in tb.results for c in sc["codes"]]
        assert {H.OK, H.BAD, H.RANGE} <= set(codes)

    def test_lease_stall_mutation_fails(self, tmp_path):
        ml = emit_mlane(CORE, 2, True)
        tb = emit_mlane_testbench(CORE, 2, True, standard_scenes(CORE, 2, True),
                                  break_stall=True)
        out = sim(tmp_path, {"lane.v": ml.lane_source, "ml.v": ml.source,
                             "tb.v": tb.source})
        assert "FAIL" in out and "PASS" not in out

    def test_lease_uart(self, tmp_path):
        ml = emit_mlane(CORE, 2, True)
        sc = uart_scenes(CORE, 2, True)
        out = sim(tmp_path, {
            "lane.v": ml.lane_source, "ml.v": ml.source,
            "top.v": emit_mlane_uart(CORE, 2, True),
            "tb.v": emit_mlane_uart_testbench(CORE, 2, True, sc)})
        assert f"PASS: {len(sc)} scenes over the UART" in out, out[-2000:]

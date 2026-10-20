"""Tests for the Verilog backend: emission checks and iverilog simulation."""

import shutil
import subprocess

import pytest

from cmtz.compiler import compile_program
from cmtz.field import IRField
from cmtz.ir_nodes import (
    IRProgram, IREmbed, IRPoly, IRMeasure, IRMatPow, IRPrimitiveRoots,
)
from cmtz.backends.verilog_backend import emit_verilog, VerilogBundle

IVERILOG = shutil.which("iverilog") and shutil.which("vvp")

REAL_MUL = """
field(17, 293);
embed(0, 1); embed(1, 5);
mul(embed_0, embed_1) as m;
rotate(m, embed_0, 3);
add(rot_m_embed_0, embed_0) as a;
measure(a, mod_p) as out;
"""

CPLX = """
cfield(7);
cembed(0, 1); cembed(1, 2);
mul(cembed_1, cembed_2) as m;
conj(m) as c;
add(m, c) as s;
magsq(s) as ms;
measure(ms, mod_p) as norm;
"""


def _simulate(bundle: VerilogBundle, tmp_path, mutate=None) -> str:
    dut = bundle.verilog_source if mutate is None else mutate(bundle.verilog_source)
    (tmp_path / f"{bundle.module_name}.v").write_text(dut)
    (tmp_path / f"{bundle.testbench_name}.v").write_text(bundle.testbench_source)
    subprocess.run(
        ["iverilog", "-g2005", "-o", "sim.out", f"{bundle.module_name}.v",
         f"{bundle.testbench_name}.v"],
        cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run(["vvp", "sim.out"], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    return out


def _bundle(src: str, **kw) -> VerilogBundle:
    result = compile_program(src, backend='python')
    assert result.ok, result.errors
    return emit_verilog(result.ir_program, **kw)


def _poly_program(coeffs, p=17, q=293) -> IRProgram:
    f = IRField(p, q)
    prog = IRProgram(ir_field=f)
    e = IREmbed(name="embed_0", ir_field=f, index=0, psi=1)
    pl = IRPoly(name="pl", ir_field=f, parents=[e], coeffs=coeffs)
    me = IRMeasure(name="out", ir_field=f, parents=[pl], mode="mod_p", src=pl)
    for n in (e, pl, me):
        prog.add_node(n)
    return prog


class TestEmission:
    def test_compile_program_backend(self):
        r = compile_program(REAL_MUL, backend='verilog')
        assert r.ok, r.errors
        out = r.backend_output
        assert out['verilog_source'].startswith("`timescale")
        assert out['testbench_source'].count("module") >= 2
        assert out['latency'] >= 1

    def test_ports_are_mul_leaves_only(self):
        b = _bundle(REAL_MUL)
        names = [pt.name for pt in b.ports]
        assert names == ["in_embed_0", "in_embed_1", "out_out"]
        assert all(pt.width == 5 for pt in b.ports)      # ceil(log2 16)

    def test_no_mul_means_constant_outputs_and_no_inputs(self):
        b = _bundle("field(17); embed(0, 3); rotate(embed_0, embed_0, 2); "
                    "measure(rot_embed_0_embed_0, mod_p) as r;")
        assert [pt.direction for pt in b.ports] == ["output"]
        assert "assign out_r = 5'd" in b.verilog_source

    def test_complex_ports_pack_two_lanes(self):
        b = _bundle(CPLX)
        assert b.ports[0].is_complex and b.ports[0].width == 6   # 2 * 3 bits
        assert b.ports[-1].width == 3

    def test_no_percent_operator(self):
        for src in (REAL_MUL, CPLX):
            assert "%" not in _bundle(src).verilog_source

    def test_inputs_override(self):
        b = _bundle(REAL_MUL, inputs=["embed_0"])
        assert [pt.name for pt in b.ports if pt.direction == "input"] == \
            ["in_embed_0"]

    def test_bad_input_name_rejected(self):
        with pytest.raises(ValueError, match="not an embed"):
            _bundle(REAL_MUL, inputs=["m"])

    def test_matpow_rejected(self):
        f = IRField(17, 293)
        prog = IRProgram(ir_field=f)
        roots = IRPrimitiveRoots(name="M", ir_field=f)
        mp = IRMatPow(name="Mp", ir_field=f, parents=[roots],
                      matrix_name="M", d=8, epsilon=0.5)
        prog.add_node(roots)
        prog.add_node(mp)
        with pytest.raises(ValueError, match="matpow"):
            emit_verilog(prog)

    def test_requires_field(self):
        with pytest.raises(ValueError, match="no field"):
            emit_verilog(IRProgram())


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestSimulation:
    @pytest.mark.parametrize("src", [REAL_MUL, CPLX], ids=["real", "complex"])
    def test_testbench_passes(self, src, tmp_path):
        assert "PASS" in _simulate(_bundle(src), tmp_path)

    def test_larger_field_random_vectors(self, tmp_path):
        src = ("cfield(31); cembed(0, 1); cembed(1, 2); "
               "mul(cembed_1, cembed_2) as m; magsq(m) as ms; "
               "measure(ms, mod_p) as norm;")
        b = _bundle(src)
        assert b.num_vectors == 66          # 2 corners + 64 random
        assert "PASS" in _simulate(b, tmp_path)

    @pytest.mark.parametrize("coeffs", [[3, 1], [5, 100, 7, 300], [0, 0, 1]])
    def test_poly_horner(self, coeffs, tmp_path):
        b = emit_verilog(_poly_program(coeffs), inputs=["embed_0"])
        assert b.latency == len(coeffs)   # degree Horner stages + final mod p
        assert "PASS" in _simulate(b, tmp_path)

    def test_testbench_detects_broken_design(self, tmp_path):
        """The bench must be able to fail: corrupt a reduction and expect FAIL."""
        b = _bundle(REAL_MUL)
        out = _simulate(b, tmp_path,
                        mutate=lambda s: s.replace("t = t - ", "t = t + ", 1))
        assert "FAIL" in out and "PASS" not in out


PHASOR_ADD = """
cfield(7);
cembed(0, 1); cembed(2, 0);
add(cembed_1, cembed_2) as sum_AB;
magsq(sum_AB) as power;
measure(power, mod_p) as combined;
"""

INTERRUPT = """
field(17, 19);
embed(0, 0); embed(1, 6);
interrupt {
    embed(2, 3);
    rotate(embed_2, embed_2, 5);
} restoring (embed_1);
measure(embed_1, mod_p) as guard_val;
measure(rot_embed_2_embed_2, mod_p) as handler_out;
"""


class TestInputsAll:
    def test_all_exposes_every_leaf_an_output_depends_on(self):
        b = _bundle(PHASOR_ADD, inputs="all")
        assert [pt.name for pt in b.ports if pt.direction == "input"] == \
            ["in_cembed_1", "in_cembed_2"]

    def test_vectors_are_lane_level(self):
        b = _bundle(PHASOR_ADD, inputs="all")
        ins, outs = b.vectors[0]
        assert len(ins) == 4 and len(outs) == 1     # two complex ports, re+im each


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestSimulationRegressions:
    def test_complex_add_keeps_imaginary_lane(self, tmp_path):
        """Add lanes are pw+1 bits wide; an unpinned concatenation shifted im."""
        assert "PASS" in _simulate(_bundle(PHASOR_ADD, inputs="all"), tmp_path)

    def test_region_inputs_follow_the_ports(self, tmp_path):
        """The reference must not re-evaluate region bodies from their literals."""
        b = _bundle(INTERRUPT, inputs="all")
        assert "PASS" in _simulate(b, tmp_path)

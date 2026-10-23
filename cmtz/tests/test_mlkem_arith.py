"""ML-KEM butterfly arithmetic: oracle sanity, exhaustive reduction ranges, RTL simulation."""
import random
import shutil
import subprocess

import pytest

from cmtz.accel.mlkem_ntt import arith, fips203_oracle as O
from cmtz.accel.mlkem_ntt.tb_bfly import emit_bfly_tb

IVERILOG = shutil.which("iverilog") and shutil.which("vvp")
Q = 3329


class TestOracle:
    def test_constants(self):
        assert O.ZETAS[1] == 1729            # FIPS 203 Appendix A, first entry
        assert (128 * 3303) % Q == 1
        assert pow(17, 128, Q) == Q - 1      # 17 has order 256

    def test_matches_defining_equation(self):
        rng = random.Random(1)
        for _ in range(2):
            f = [rng.randrange(Q) for _ in range(256)]
            assert O.ntt(f) == O.direct_ntt(f)

    def test_inverse_roundtrip(self):
        rng = random.Random(2)
        for _ in range(5):
            f = [rng.randrange(Q) for _ in range(256)]
            assert O.ntt_inv(O.ntt(f)) == f
            assert O.ntt(O.ntt_inv(f)) == f


class TestReductionRange:
    """Exhaustive over the whole supported domain: p = x*z for canonical x, z."""

    @pytest.mark.parametrize("reduction,rinv", [("montgomery", pow(arith.R_MONT, -1, Q)),
                                                 ("barrett", 1)])
    def test_exhaustive(self, reduction, rinv):
        red = arith.reduce_fn(reduction)
        for p in range((Q - 1) ** 2 + 1):
            u = red(p)
            assert 0 <= u < 2 * Q
            assert arith.cond_sub(u) == (p * rinv) % Q, p


class TestButterflyModel:
    @pytest.mark.parametrize("reduction", ["montgomery", "barrett"])
    def test_against_spec_formulas(self, reduction):
        rng = random.Random(3)
        edge = [0, 1, 2, Q // 2, Q - 2, Q - 1]
        cases = [(a, b, k) for a in edge for b in edge for k in range(0, 128, 9)]
        cases += [(rng.randrange(Q), rng.randrange(Q), rng.randrange(128)) for _ in range(20000)]
        for a, b, k in cases:
            zeta = O.ZETAS[k]
            z = arith.twiddle_encode(reduction, zeta, False)
            t = (b * zeta) % Q
            assert arith.bfly_model(reduction, 0, a, b, z) == ((a + t) % Q, (a - t) % Q)
            z = arith.twiddle_encode(reduction, zeta, True)
            # the model's per-layer halving reproduces (spec value) * 2^-1
            exp0 = ((a + b) * pow(2, -1, Q)) % Q
            exp1 = (zeta * (b - a) * pow(2, -1, Q)) % Q
            assert arith.bfly_model(reduction, 1, a, b, z) == (exp0, exp1)


def _sim(reduction, tmp_path, stride, depth="deep"):
    (tmp_path / "bfly.v").write_text(arith.emit_butterfly(reduction, depth))
    (tmp_path / "bfly_tb.v").write_text(emit_bfly_tb(reduction, depth=depth))
    subprocess.run(["iverilog", "-g2012", f"-DSTRIDE={stride}", "-o", "s.out", "bfly.v",
                    "bfly_tb.v"], cwd=tmp_path, check=True, capture_output=True)
    return subprocess.run(["vvp", "s.out"], cwd=tmp_path, check=True, capture_output=True,
                          text=True).stdout


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestButterflySimulation:
    @pytest.mark.parametrize("depth", ["deep", "compact"])
    @pytest.mark.parametrize("reduction", ["montgomery", "barrett"])
    def test_sampled_sweep(self, reduction, depth, tmp_path):
        out = _sim(reduction, tmp_path, 211, depth)
        assert f"PASS mlkem_bfly {reduction}" in out, out

    def test_testbench_detects_a_broken_design(self, tmp_path):
        src = arith.emit_butterfly("montgomery").replace("(p5[15:0] << 8) - p5[15:0]",
                                                         "(p5[15:0] << 8) - p5[15:0] - 1")
        (tmp_path / "bfly.v").write_text(src)
        (tmp_path / "bfly_tb.v").write_text(emit_bfly_tb("montgomery"))
        subprocess.run(["iverilog", "-g2012", "-DSTRIDE=211", "-o", "s.out", "bfly.v",
                        "bfly_tb.v"], cwd=tmp_path, check=True, capture_output=True)
        out = subprocess.run(["vvp", "s.out"], cwd=tmp_path, capture_output=True, text=True).stdout
        assert "FAIL" in out and "PASS" not in out

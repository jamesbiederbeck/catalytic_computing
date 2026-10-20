"""Tests for the hardware view and the visualizer server's hardware routes."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from cmtz.compiler import compile_program
from cmtz.hwview import HardwareView, hardware_manifest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _view(name: str) -> HardwareView:
    r = compile_program((EXAMPLES / name).read_text())
    assert r.ok, r.errors
    return HardwareView(r.ir_program)


class TestHardwareView:
    def test_manifest_shape(self):
        m = _view("fibonacci_pure.cmtz").manifest()
        assert m["ok"] and m["p"] == 17 and m["latency"] == 10
        assert (m["request_bytes"], m["reply_bytes"], m["state_bytes"]) == (2, 1, 12)
        assert len(m["hash_hex"]) == 8
        kinds = {n["name"]: n["kind"] for n in m["nodes"]}
        assert kinds["embed_0"] == "input" and kinds["F12"] == "reg"
        assert kinds["result"] == "alias"

    def test_evaluate_matches_the_fibonacci_recurrence(self):
        e = _view("fibonacci_pure.cmtz").evaluate([1, 1])
        assert e["reply"] == [8]                          # F(12) mod 17
        assert e["state"] == [1, 1, 2, 3, 5, 8, 13, 4, 0, 4, 4, 8]
        assert e["values"]["F12"] == 8

    def test_evaluate_complex_lanes(self):
        e = _view("07_phasor_interference.cmtz").evaluate([6, 3, 6, 3])
        # sum = (5, 6); |sum|^2 = 25 - 36*3 = 1 mod 7
        assert e["reply"][0] == 1
        assert e["values"]["sum_AB"] == [5, 6]
        assert e["state"][:4] == [6, 3, 6, 3]

    @pytest.mark.parametrize("lanes", [[1], [1, 2, 3], [1, 17], [-1, 0], [1, "2"]])
    def test_evaluate_rejects_bad_lanes(self, lanes):
        with pytest.raises(ValueError):
            _view("fibonacci_pure.cmtz").evaluate(lanes)

    def test_matpow_program_is_reported_not_raised(self):
        r = compile_program((EXAMPLES / "fibonacci.cmtz").read_text())
        m = hardware_manifest(r.ir_program)
        assert m["ok"] is False and "matpow" in m["error"]


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location(
        "viz_server", ROOT / "visualizer" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    httpd = HTTPServer(("127.0.0.1", 0), mod.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


class TestServerRoutes:
    SRC = (EXAMPLES / "fibonacci_pure.cmtz").read_text()

    def test_compile_includes_hardware_manifest(self, server):
        status, d = _post(server, "/api/compile", {"source": self.SRC})
        assert status == 200 and d["ok"] and d["hw"]["ok"]
        assert d["hw"]["latency"] == 10

    def test_compile_still_works_when_hardware_is_unavailable(self, server):
        src = (EXAMPLES / "fibonacci.cmtz").read_text()
        status, d = _post(server, "/api/compile", {"source": src})
        assert status == 200 and d["ok"] and d["hw"]["ok"] is False

    def test_hw_eval(self, server):
        status, d = _post(server, "/api/hw/eval",
                          {"source": self.SRC, "lanes": [1, 1]})
        assert status == 200 and d["reply"] == [8]

    def test_hw_eval_bad_lanes_is_400(self, server):
        status, d = _post(server, "/api/hw/eval",
                          {"source": self.SRC, "lanes": [1, 99]})
        assert status == 400 and d["ok"] is False

    def test_hw_eval_without_hardware_is_422(self, server):
        src = (EXAMPLES / "fibonacci.cmtz").read_text()
        status, d = _post(server, "/api/hw/eval", {"source": src, "lanes": []})
        assert status == 422 and "matpow" in d["error"]

    def test_static_js_served_only_from_visualizer_dir(self, server):
        for path in ("/../server.py", "/nope.js", "/..%2fserver.py"):
            try:
                urllib.request.urlopen(server + path)
                assert False, path
            except urllib.error.HTTPError as e:
                assert e.code == 404


# ── golden fixture shared with the JS tests (visualizer/board.test.mjs) ───────

import os

from cmtz.backends.verilog_uart import info_frame

GOLDEN = ROOT / "visualizer" / "testdata" / "golden.json"
GOLDEN_CASES = {
    "fibonacci_pure.cmtz": [[1, 1], [16, 16], [3, 5], [0, 9]],
    "07_phasor_interference.cmtz": [[6, 3, 6, 3], [0, 0, 0, 0], [1, 2, 3, 4]],
}


def _golden() -> dict:
    out = {}
    for name, cases in GOLDEN_CASES.items():
        hv = _view(name)
        m = hv.manifest()
        out[name] = {
            "manifest": m,
            "info_bytes": info_frame(hv.bundle, m["request_bytes"],
                                     m["reply_bytes"], m["state_bytes"]),
            "cases": [{"lanes": lanes, **{k: v for k, v in hv.evaluate(lanes).items()
                                          if k in ("reply", "state", "values")}}
                      for lanes in cases],
        }
    return out


def test_golden_fixture_is_current():
    """The JS tests replay this file; regenerate with UPDATE_GOLDEN=1."""
    fresh = json.loads(json.dumps(_golden()))
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    assert json.loads(GOLDEN.read_text()) == fresh

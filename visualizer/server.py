#!/usr/bin/env python3
"""
Cook-Mertz IR Visualizer — HTTP server.

Serves the SPA and exposes a /api/compile endpoint.
Binds to 0.0.0.0 so the whole LAN can reach it.

Usage:
    cd /home/victor/code/cook_mertz
    uv run python visualizer/server.py [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cmtz import compile_program
from cmtz.field import IRField, primitive_roots_Fp
from cmtz.ir_nodes import (
    IREmbed, IRComplexEmbed, IRRotate, IRAdd, IRCompose, IRMeasure,
    IRPrimitiveRoots, IRConjugate, IRMagnitudeSq, IRMatPow,
    IRRegion, IRCatalyticRegion, IRCycloPhiPoly, IRPoly,
)
from cmtz.backends.python_ref import PythonBackend

HERE = Path(__file__).parent


# ── Phase-shifted backend ─────────────────────────────────────────────────────

class PhasedBackend(PythonBackend):
    """Re-evaluates a compiled IR program with all embed phases shifted by k."""

    def __init__(self, ir_field: IRField, offset: int):
        super().__init__(ir_field)
        self.offset = offset
        self._m = len(self.roots)

    def eval_embed(self, node: IREmbed) -> int:
        return self.roots[(node.psi + self.offset) % self._m]

    def eval_complex_embed(self, node: IRComplexEmbed) -> tuple[int, int]:
        m = self._m
        return (
            self.roots[(node.re_psi + self.offset) % m],
            self.roots[(node.im_psi + self.offset) % m],
        )


def compute_frames(ir_prog, ir_field: IRField | None) -> tuple[list[dict], int]:
    """Return (frames, m) where frames[k] maps name→{num, str} for phase k."""
    if ir_field is None:
        return [], 0

    m = ir_field.p - 1  # order of F_p^*

    frames = []
    for k in range(m):
        be = PhasedBackend(ir_field, k)
        vals = be.eval_program(ir_prog)
        frame: dict[str, dict] = {}
        for name, val in vals.items():
            if isinstance(val, tuple):
                frame[name] = {"str": f"({val[0]},{val[1]})", "num": list(val)}
            else:
                frame[name] = {"str": str(val), "num": val}
        frames.append(frame)

    return frames, m


# ── Compiler → JSON serialisation ────────────────────────────────────────────

def field_to_dict(f) -> dict | None:
    if f is None:
        return None
    return {"p": f.p, "q": f.q, "c": f.c, "is_complex": f.is_complex}


def node_details(node) -> dict:
    d = {}
    if isinstance(node, IREmbed):
        d["psi"] = node.psi
        d["index"] = node.index
    elif isinstance(node, IRComplexEmbed):
        d["re_psi"] = node.re_psi
        d["im_psi"] = node.im_psi
    elif isinstance(node, IRRotate):
        d["j"] = node.j
    elif isinstance(node, IRMatPow):
        d["d"] = node.d
        d["epsilon"] = node.epsilon
        d["delta"] = int(node.delta)
        cb = node.cost_bound()
        d["budget_recursive"] = round(cb["recursive_calls"], 2)
        d["budget_instrs"] = round(cb["basic_instructions"], 2)
    elif isinstance(node, IRMeasure):
        d["mode"] = node.mode
    elif isinstance(node, (IRRegion, IRCatalyticRegion)):
        d["inner"] = [n.name for n in node.inner_nodes]
        d["catalytic_regs"] = node.catalytic_regs
        d["strict"] = isinstance(node, IRCatalyticRegion)
    return d


def assign_layers(nodes_list: list) -> dict[str, int]:
    name_to_node = {name: node for name, node in nodes_list}
    layer: dict[str, int] = {}

    def depth(name: str) -> int:
        if name in layer:
            return layer[name]
        node = name_to_node[name]
        if not node.parents:
            layer[name] = 0
        else:
            layer[name] = 1 + max(depth(p.name) for p in node.parents)
        return layer[name]

    for name, _ in nodes_list:
        depth(name)
    return layer


def compile_to_graph(source: str) -> dict:
    result = compile_program(source, backend="python", optimize=True)

    if not result.ok:
        return {"ok": False, "errors": result.errors, "nodes": [],
                "analysis": {}, "optimization": {}, "frames": [], "m": 0}

    prog = result.ir_program
    nodes_list = list(prog.nodes())
    layers = assign_layers(nodes_list)
    backend_out = result.backend_output or {}

    nodes_json = []
    for name, node in nodes_list:
        val = backend_out.get(name)
        if isinstance(val, tuple):
            val_str = f"({val[0]},{val[1]})"
            val_num  = list(val)
        elif val is not None:
            val_str = str(val)
            val_num  = val
        else:
            val_str = None
            val_num  = None

        nodes_json.append({
            "name":        name,
            "type":        type(node).__name__,
            "parents":     [p.name for p in node.parents],
            "field":       field_to_dict(node.ir_field),
            "value":       val_str,
            "value_num":   val_num,
            "details":     node_details(node),
            "cost":        {
                "t": node.recursive_calls,
                "s": node.basic_instructions,
                "r": node.num_registers,
            },
            "layer":       layers.get(name, 0),
            "is_catalytic": getattr(node, "is_catalytic", False),
        })

    frames, m = compute_frames(prog, prog.ir_field)

    return {
        "ok":           True,
        "errors":       [],
        "nodes":        nodes_json,
        "analysis":     result.analysis_report,
        "optimization": result.optimization_stats,
        "frames":       frames,
        "m":            m,
    }


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_file(HERE / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/examples":
            examples_dir = HERE.parent / "examples"
            files = sorted(f.name for f in examples_dir.glob("*.cmtz"))
            self.send_json({"examples": files})
        elif self.path.startswith("/api/example/"):
            fname = self.path[len("/api/example/"):]
            fpath = HERE.parent / "examples" / fname
            if fpath.exists() and fpath.suffix == ".cmtz":
                self.send_json({"source": fpath.read_text()})
            else:
                self.send_json({"error": "not found"}, 404)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/compile":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
                source = payload.get("source", "")
                graph = compile_to_graph(source)
                self.send_json(graph)
            except Exception as e:
                self.send_json({"ok": False, "errors": [str(e)], "nodes": [],
                                "frames": [], "m": 0}, 500)
        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    ip = local_ip()
    server = HTTPServer(("0.0.0.0", args.port), Handler)

    print(f"\nCook-Mertz IR Visualizer")
    print(f"  Local :  http://localhost:{args.port}")
    print(f"  LAN   :  http://{ip}:{args.port}")
    print(f"\nCtrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

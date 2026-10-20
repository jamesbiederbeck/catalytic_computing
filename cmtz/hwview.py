"""Hardware view of a compiled program, for the browser visualizer.

Wraps the Verilog backend's debug bundle: a manifest the page draws (what the
FPGA does with each IR node, and the UART lane layout) and a reference
evaluation under given inputs (what the board should answer, and what its
state readback should hold). No board access.
"""

from __future__ import annotations

from typing import Any

from .backends.verilog_backend import emit_verilog, evaluate_all
from .backends.verilog_uart import emit_uart_wrapper, lane_table, dbg_layout
from .ir_nodes import IRProgram


def _jsonable(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def _lanes_of(v: Any) -> list[int]:
    return list(v) if isinstance(v, tuple) else [v]


class HardwareView:
    """The debug-protocol hardware lowering of `prog` (inputs="all")."""

    def __init__(self, prog: IRProgram):
        if prog.ir_field is None:
            raise ValueError("Program has no field annotation")
        self.prog = prog
        self.field = prog.ir_field
        self.bundle = emit_verilog(prog, module_name="cmtz_top",
                                   inputs="all", debug=True)
        self.wrapper = emit_uart_wrapper(self.bundle)
        self.lanes = lane_table(self.bundle)

    def manifest(self) -> dict:
        b, f = self.bundle, self.field
        h = b.program_hash
        return {
            "ok": True,
            "p": f.p, "q": f.q, "c": f.c, "complex": f.is_complex,
            "latency": b.latency,
            "hash": h, "hash_hex": f"{h:08x}",
            "request_bytes": self.wrapper.request_bytes,
            "reply_bytes": self.wrapper.reply_bytes,
            "state_bytes": len(self.lanes["state"]),
            "lanes": self.lanes,
            "nodes": [
                {"name": n.name, "op": n.op, "kind": n.kind, "depth": n.depth,
                 "width": n.width, "complex": n.is_complex,
                 "const": _jsonable(n.const)}
                for n in b.nodes],
        }

    def evaluate(self, lanes_in: list[int]) -> dict:
        """Reference answers for one request (`lanes_in` in lane order)."""
        want = self.wrapper.request_bytes
        if len(lanes_in) != want:
            raise ValueError(f"expected {want} input lanes, got {len(lanes_in)}")
        p = self.field.p
        if any(not isinstance(x, int) or not 0 <= x < p for x in lanes_in):
            raise ValueError(f"input lanes must be integers in 0..{p - 1}")

        overrides: dict[str, Any] = {}
        for slot, x in zip(self.lanes["inputs"], lanes_in):
            if slot["lane"] == "":
                overrides[slot["node"]] = x
            else:
                cur = list(overrides.get(slot["node"], (0, 0)))
                cur[0 if slot["lane"] == "re" else 1] = x
                overrides[slot["node"]] = tuple(cur)
        regs = evaluate_all(self.prog, self.field, overrides)

        def lane(node: str, which: str) -> int:
            v = regs[node]
            return v[1] if which == "im" else (v[0] if which == "re" else v)

        return {
            "values": {k: _jsonable(v) for k, v in regs.items()
                       if not isinstance(v, (list, dict))},
            "reply": [lane(s["node"], s["lane"]) for s in self.lanes["outputs"]],
            "state": [lane(n, w) for n, w, _ in dbg_layout(self.bundle)],
        }


def hardware_manifest(prog: IRProgram) -> dict:
    """Manifest, or {"ok": False, "error": ...} if the program cannot be
    lowered (matpow, no inputs, p > 255, ...)."""
    try:
        return HardwareView(prog).manifest()
    except ValueError as e:
        return {"ok": False, "error": str(e)}

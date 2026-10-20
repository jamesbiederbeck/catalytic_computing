"""Verilog backend for Cook-Mertz v4.

Lowers an IR DAG to a feed-forward, fully pipelined Verilog-2001 module plus a
self-checking testbench. Plain Verilog on purpose: iverilog and yosys (the
open-source Tang Nano flow) reject much of SystemVerilog.

Design:
  - No FSM. Every runtime IR node owns a registered output, so throughput is
    one result per clock and latency is the DAG depth. A `valid` bit shifts
    alongside the data.
  - Only runtime values cost hardware. Embeds, literal-`j` rotations and
    literal-coefficient polys of constants are folded at emit time by the
    Python reference backend, so the two cannot disagree. A rotation of a
    runtime value is a constant-coefficient multiply mod p (no ROM).
  - Input ports default to the leaf embed / cembed nodes feeding an IRMul
    (the only IR primitive whose operands are both runtime); `inputs=`
    overrides. The embed constant is then only documentation.
  - Widths are per node: mod-p nodes use ceil(log2 p) bits per lane, IRPoly
    accumulates mod q.
  - No `%`: yosys builds a full divider for it even with a constant modulus.
    Reductions are explicit unrolled conditional subtracts of p << k.
  - IRMatPow is rejected. IRRegion / IRCatalyticRegion are comments only;
    catalytic restoration is a verification property, not synthesised logic.

Complex (F_{p^2}) values travel as one vector `{im, re}`.
"""

from __future__ import annotations

import itertools
import json
import random
import zlib
from dataclasses import dataclass, field
from typing import Optional, Union

from ..ir_nodes import (
    IRProgram, IRNode, IREmbed, IRPrimitiveRoots, IRRotate, IRPoly, IRMatPow,
    IRMeasure, IRCompose, IRRegion, IRCatalyticRegion, IRCycloPhiPoly, IRAdd,
    IRComplexEmbed, IRConjugate, IRMagnitudeSq, IRMul,
)
from ..field import IRField, primitive_roots_Fp
from .python_ref import PythonBackend


# ── Output types ─────────────────────────────────────────────────────────────

@dataclass
class VerilogPort:
    name: str
    direction: str          # 'input' | 'output'
    width: int
    node: Optional[str] = None      # IR node the port carries
    is_complex: bool = False


@dataclass
class NodeInfo:
    """What the hardware does with one IR node (for visualisation and state
    readback)."""
    name: str
    op: str                         # IR node class name
    parents: list[str]
    kind: str                       # 'const' | 'input' | 'reg' | 'alias'
    depth: int                      # pipeline stage; 0 for ports and constants
    width: int                      # bits of the value (both lanes if complex)
    is_complex: bool
    const: Union[int, tuple[int, int], None] = None
    dbg_offset: Optional[int] = None    # bit offset in dbg_state, if exposed


@dataclass
class VerilogBundle:
    """Complete output from the Verilog backend."""
    verilog_source: str
    testbench_source: str
    module_name: str
    ports: list[VerilogPort]
    latency: int            # clocks from in_valid to out_valid
    field_p: int
    field_q: int
    num_vectors: int
    # (input lanes, output lanes) per test vector, in port order; a complex
    # value contributes two lanes, re then im. Shared with the UART wrapper
    # testbench so both check the same expected values.
    vectors: list[tuple[list[int], list[int]]] = field(default_factory=list)
    nodes: list[NodeInfo] = field(default_factory=list)
    debug: bool = False             # datapath has a dbg_state output
    dbg_width: int = 0
    # per vector, the dbg lanes (one per lane of each exposed node, in
    # `nodes` order) the datapath should hold once it settles; debug only
    dbg_vectors: list[list[int]] = field(default_factory=list)
    program_hash: int = 0           # crc32 of the hardware structure

    @property
    def testbench_name(self) -> str:
        return f"{self.module_name}_tb"

    def simulate_command(self) -> str:
        """iverilog command for the generated design + testbench."""
        return (f"iverilog -o sim.out {self.module_name}.v "
                f"{self.testbench_name}.v && vvp sim.out")


# ── Internal value tracking ──────────────────────────────────────────────────

@dataclass
class _Val:
    """A node's value: an emit-time constant or a runtime signal."""
    cplx: bool
    const: Union[int, tuple[int, int], None] = None
    sig: str = ""
    depth: int = 0          # clocks after the ports; 0 = combinational port
    width: int = 0

    @property
    def is_const(self) -> bool:
        return self.const is not None


def _bits(n: int) -> int:
    return max(1, n.bit_length())


def _lit(width: int, value: int) -> str:
    return f"{width}'d{value}"


# ── Emitter ──────────────────────────────────────────────────────────────────

class VerilogEmitter:
    def __init__(self, ir_field: IRField, module_name: str = "cmtz_top",
                 inputs: Union[list[str], str, None] = None,
                 num_vectors: int = 64, seed: int = 0, debug: bool = False):
        self.debug = debug
        self.field = ir_field
        self.p = ir_field.p
        self.q = ir_field.q
        self.module_name = module_name
        self.requested_inputs = inputs
        self.num_vectors = num_vectors
        self.seed = seed
        self.pw = _bits(self.p - 1)

        self.roots = primitive_roots_Fp(self.p)
        self._ref = PythonBackend(ir_field)     # folds constants
        self._vals: dict[str, _Val] = {}
        self._decls: list[str] = []
        self._body: list[str] = []
        self._funcs: dict[str, str] = {}
        self._delays: set[str] = set()

    # ── Verilog snippets ─────────────────────────────────────────────

    def _delay(self, v: _Val, to_depth: int) -> str:
        """Signal name of `v` delayed to pipeline depth `to_depth`."""
        if to_depth < v.depth:
            raise ValueError("internal: cannot delay to an earlier depth")
        prev = v.sig
        for n in range(1, to_depth - v.depth + 1):
            name = f"{v.sig}_z{n}"
            if name not in self._delays:
                self._delays.add(name)
                self._decls.append(f"reg [{v.width - 1}:0] {name};")
                self._body.append(
                    f"always @(posedge clk) {name} <= {prev};")
            prev = name
        return prev

    def _lanes(self, v: _Val, depth: int) -> tuple[str, Optional[str]]:
        """(re, im) Verilog expressions for `v` aligned to `depth`."""
        pw = self.pw
        if v.is_const:
            if v.cplx:
                re, im = v.const
                return _lit(pw, re), _lit(pw, im)
            return _lit(pw, v.const), None
        sig = self._delay(v, depth)
        if v.cplx:
            return f"{sig}[{pw - 1}:0]", f"{sig}[{2 * pw - 1}:{pw}]"
        return sig, None

    def _wire(self, name: str, width: int, expr: str) -> str:
        self._decls.append(f"wire [{width - 1}:0] {name} = {expr};")
        return name

    def _reduce(self, pre: str, xmax: int, mod: int) -> str:
        """Expression reducing `pre` (an xw-bit wire, value <= xmax) mod `mod`."""
        xw = _bits(xmax)
        ow = _bits(mod - 1)
        fname = f"red_{mod}_{xw}"
        if fname not in self._funcs:
            lines = [f"function [{ow - 1}:0] {fname};",
                     f"    input [{xw - 1}:0] x;",
                     f"    reg [{xw - 1}:0] t;",
                     "    begin",
                     "        t = x;"]
            if xmax >= mod:
                kmax = 0
                while (mod << (kmax + 1)) <= xmax:
                    kmax += 1
                for k in range(kmax, -1, -1):
                    m = mod << k
                    lines.append(
                        f"        if (t >= {_lit(xw, m)}) t = t - {_lit(xw, m)};")
            lines += [f"        {fname} = t[{ow - 1}:0];",
                      "    end",
                      "endfunction"]
            self._funcs[fname] = "\n".join(lines)
        return f"{fname}({pre})"

    def _stage(self, name: str, depth_in: int, lanes: list[str]) -> _Val:
        """Register one stage: `lanes` are next-value expressions (re[, im])."""
        pw = self.pw
        cplx = len(lanes) == 2
        width = 2 * pw if cplx else pw
        # Concatenation uses each operand's own width, so pin every lane to
        # exactly pw bits first (an add lane is pw+1 bits wide).
        lw = [self._wire(f"n_{name}_{tag}_nx", pw, e)
              for tag, e in zip(("re", "im"), lanes)]
        nx = "{" + lw[1] + ", " + lw[0] + "}" if cplx else lw[0]
        sig = f"n_{name}"
        self._decls.append(f"reg [{width - 1}:0] {sig};")
        self._body.append(f"always @(posedge clk) {sig} <= {nx};")
        return _Val(cplx=cplx, sig=sig, depth=depth_in + 1, width=width)

    # ── Node lowering ────────────────────────────────────────────────

    def _runtime_depth(self, *vals: _Val) -> int:
        return max((v.depth for v in vals if not v.is_const), default=0)

    def _fold(self, node: IRNode, parents: list[_Val]) -> _Val:
        """Evaluate `node` with the reference backend (all parents constant)."""
        for par, v in zip(node.parents, parents):
            self._ref.registers[par.name] = v.const
        val = self._ref._eval_node(node)
        self._ref.registers[node.name] = val
        return _Val(cplx=isinstance(val, tuple), const=val)

    def _lower_node(self, node: IRNode) -> _Val:
        name = node.name
        pw, p = self.pw, self.p
        parents = [self._vals[par.name] for par in node.parents
                   if par.name in self._vals]

        if isinstance(node, IRMatPow):
            raise ValueError(
                f"matpow '{name}' is not supported by the Verilog backend")
        if isinstance(node, (IRRegion, IRCatalyticRegion)):
            # Regions are not hardware; like the reference backend, a region's
            # value is its last inner node's value.
            self._body.append(f"// {type(node).__name__} '{name}': "
                              f"restoring {node.catalytic_regs}, not lowered")
            if node.inner_nodes and node.inner_nodes[-1].name in self._vals:
                return self._vals[node.inner_nodes[-1].name]
            return _Val(cplx=False, const=0)
        if isinstance(node, (IRPrimitiveRoots, IRCycloPhiPoly)):
            self._body.append(f"// {type(node).__name__} '{name}': "
                              f"not lowered to hardware")
            return _Val(cplx=False, const=0)

        if isinstance(node, (IREmbed, IRComplexEmbed)):
            if name in self._input_names:
                cplx = isinstance(node, IRComplexEmbed)
                return _Val(cplx=cplx, sig=f"in_{name}", depth=0,
                            width=2 * pw if cplx else pw)
            return self._fold(node, parents)

        if isinstance(node, IRCompose):
            return self._vals[node.parents[-1].name] if node.parents \
                else _Val(cplx=False, const=0)
        if isinstance(node, IRMeasure):
            src = parents[0]
            if src.cplx:
                raise ValueError(f"measure '{name}': complex source "
                                 f"unsupported (take magsq first)")
            return src      # values are already in [0, p)

        if all(v.is_const for v in parents) and not isinstance(node, IRPoly):
            return self._fold(node, parents)

        if isinstance(node, IRRotate):
            return self._lower_rotate(node, parents[0])
        if isinstance(node, IRPoly):
            return self._lower_poly(node, parents[0])
        if isinstance(node, IRAdd):
            return self._lower_add(node, *parents)
        if isinstance(node, IRMul):
            return self._lower_mul(node, *parents)
        if isinstance(node, IRConjugate):
            return self._lower_conj(node, parents[0])
        if isinstance(node, IRMagnitudeSq):
            return self._lower_magsq(node, parents[0])
        raise ValueError(f"{type(node).__name__} '{name}' unsupported")

    def _nonres(self, node: IRNode, parent: Optional[IRNode] = None) -> int:
        for f in (node.ir_field, parent.ir_field if parent else None,
                  self.field):
            if f is not None and f.c is not None:
                return f.c
        raise ValueError(f"'{node.name}': complex operation needs a "
                         f"non-residue c on the field")

    def _lower_rotate(self, node: IRRotate, x: _Val) -> _Val:
        if x.cplx:
            raise ValueError(f"rotate '{node.name}': complex input unsupported")
        pw, p = self.pw, self.p
        w = self.roots[node.j % len(self.roots)]
        a, _ = self._lanes(x, x.depth)
        if w == 1:
            return self._stage(node.name, x.depth, [a])
        xmax = (p - 1) * w
        pre = self._wire(f"{node.name}_pre", _bits(xmax),
                         f"{a} * {_lit(_bits(xmax), w)}")
        return self._stage(node.name, x.depth, [self._reduce(pre, xmax, p)])

    def _lower_add(self, node: IRAdd, x: _Val, y: _Val) -> _Val:
        if x.cplx != y.cplx:
            raise ValueError(f"add '{node.name}': mixed real/complex")
        pw, p = self.pw, self.p
        d = self._runtime_depth(x, y)
        xl, yl = self._lanes(x, d), self._lanes(y, d)
        lanes = []
        for i, tag in enumerate(("re", "im")[:2 if x.cplx else 1]):
            s = self._wire(f"{node.name}_{tag}_s", pw + 1, f"{xl[i]} + {yl[i]}")
            lanes.append(f"({s} >= {_lit(pw + 1, p)}) ? "
                         f"({s} - {_lit(pw + 1, p)}) : {s}")
        return self._stage(node.name, d, lanes)

    def _lower_mul(self, node: IRMul, x: _Val, y: _Val) -> _Val:
        if x.cplx != y.cplx:
            raise ValueError(f"mul '{node.name}': mixed real/complex")
        p = self.p
        d = self._runtime_depth(x, y)
        (a, b), (c_, dd) = self._lanes(x, d), self._lanes(y, d)
        if not x.cplx:
            xmax = (p - 1) ** 2
            pre = self._wire(f"{node.name}_pre", _bits(xmax), f"{a} * {c_}")
            return self._stage(node.name, d, [self._reduce(pre, xmax, p)])
        c = self._nonres(node)
        re_max = (p - 1) ** 2 * (1 + c)
        im_max = 2 * (p - 1) ** 2
        re = self._wire(f"{node.name}_re_pre", _bits(re_max),
                        f"{a} * {c_} + {b} * {dd} * {_lit(_bits(re_max), c)}")
        im = self._wire(f"{node.name}_im_pre", _bits(im_max),
                        f"{a} * {dd} + {b} * {c_}")
        return self._stage(node.name, d, [self._reduce(re, re_max, p),
                                          self._reduce(im, im_max, p)])

    def _lower_conj(self, node: IRConjugate, x: _Val) -> _Val:
        pw, p = self.pw, self.p
        re, im = self._lanes(x, x.depth)
        neg = (f"({im} == {_lit(pw, 0)}) ? {_lit(pw, 0)} : "
               f"({_lit(pw + 1, p)} - {im})")
        return self._stage(node.name, x.depth, [re, neg])

    def _lower_magsq(self, node: IRMagnitudeSq, x: _Val) -> _Val:
        # a^2 - b^2 c  ==  a^2 + b^2 (p - c)  (mod p), avoiding subtraction
        p = self.p
        c = self._nonres(node, node.parents[0])
        a, b = self._lanes(x, x.depth)
        xmax = (p - 1) ** 2 * (1 + (p - c))
        pre = self._wire(f"{node.name}_pre", _bits(xmax),
                         f"{a} * {a} + {b} * {b} * {_lit(_bits(xmax), p - c)}")
        return self._stage(node.name, x.depth, [self._reduce(pre, xmax, p)])

    def _lower_poly(self, node: IRPoly, x: _Val) -> _Val:
        """Horner mod q, one register stage per coefficient, then reduce mod p."""
        f = node.ir_field or self.field
        p, q = f.p, f.q
        if x.cplx:
            raise ValueError(f"poly '{node.name}': complex input unsupported")
        coeffs = node.coeffs
        if len(coeffs) <= 1 or x.is_const:
            return self._fold(node, [x])
        pw, qw = self.pw, _bits(q - 1)
        acc: Union[int, str] = coeffs[-1] % q       # int constant, then signal
        depth = x.depth
        for i, c in enumerate(reversed(coeffs[:-1])):
            xs = self._delay(x, depth)
            xmax = (q - 1) * (p - 1) + (q - 1)
            xw = _bits(xmax)
            a = _lit(xw, acc) if isinstance(acc, int) else acc
            pre = self._wire(f"{node.name}_h{i}_pre", xw,
                             f"{a} * {xs} + {_lit(xw, c % q)}")
            sig = f"n_{node.name}_h{i}"
            self._decls.append(f"reg [{qw - 1}:0] {sig};")
            self._body.append(f"always @(posedge clk) {sig} <= "
                              f"{self._reduce(pre, xmax, q)};")
            acc, depth = sig, depth + 1
        # acc < q; final reduction mod p (width qw into the reducer)
        fin = self._wire(f"{node.name}_fin", qw, str(acc))
        out = f"n_{node.name}"
        self._decls.append(f"reg [{pw - 1}:0] {out};")
        self._body.append(f"always @(posedge clk) {out} <= "
                          f"{self._reduce(fin, q - 1, p)};")
        return _Val(cplx=False, sig=out, depth=depth + 1, width=pw)

    # ── Program-level generation ─────────────────────────────────────

    def _choose_inputs(self, prog: IRProgram) -> list[str]:
        leaves = [n for n, node in prog.nodes()
                  if isinstance(node, (IREmbed, IRComplexEmbed))]

        def walk(node: IRNode, seen: set[str]) -> None:
            for par in node.parents:
                if par.name not in seen:
                    seen.add(par.name)
                    walk(par, seen)

        if self.requested_inputs == "all":
            # every leaf embed that some output depends on
            anc: set[str] = set()
            for out in self._choose_outputs(prog):
                anc.add(out)
                walk(prog.get_node(out), anc)
            return [n for n in leaves if n in anc]
        if self.requested_inputs is not None:
            for n in self.requested_inputs:
                if n not in leaves:
                    raise ValueError(f"input '{n}' is not an embed/cembed node")
            return list(self.requested_inputs)
        feeding: set[str] = set()
        for _, node in prog.nodes():
            if isinstance(node, IRMul):
                feeding.add(node.name)
                walk(node, feeding)
        return [n for n in leaves if n in feeding]

    def _choose_outputs(self, prog: IRProgram) -> list[str]:
        measures = [n for n, node in prog.nodes()
                    if isinstance(node, IRMeasure)]
        if measures:
            return measures
        used = {par.name for _, node in prog.nodes() for par in node.parents}
        skip = (IRPrimitiveRoots, IRRegion, IRCatalyticRegion, IRCycloPhiPoly)
        return [n for n, node in prog.nodes()
                if n not in used and not isinstance(node, skip)]

    def generate(self, prog: IRProgram) -> VerilogBundle:
        pw = self.pw
        order = prog.topo_order()
        self._input_names = self._choose_inputs(prog)
        out_names = self._choose_outputs(prog)

        for name, node in order:
            self._vals[name] = self._lower_node(node)

        ports: list[VerilogPort] = []
        for n in self._input_names:
            v = self._vals[n]
            ports.append(VerilogPort(f"in_{n}", "input", v.width, n, v.cplx))

        runtime_outs = [self._vals[n] for n in out_names
                        if not self._vals[n].is_const]
        latency = max([1] + [v.depth for v in runtime_outs])

        assigns = []
        for n in out_names:
            v = self._vals[n]
            width = 2 * pw if v.cplx else pw
            ports.append(VerilogPort(f"out_{n}", "output", width, n, v.cplx))
            if v.is_const:
                c = v.const
                val = (c[0] | (c[1] << pw)) if v.cplx else c
                assigns.append(f"assign out_{n} = {_lit(width, val)};")
            else:
                assigns.append(f"assign out_{n} = {self._delay(v, latency)};")

        nodes, dbg_width = self._manifest(order)
        src = self._render(ports, assigns, latency, out_names, nodes, dbg_width)
        tb, nvec, lanes, dbg_vecs = self._testbench(
            prog, ports, latency, out_names, nodes)
        key = json.dumps({
            "p": self.p, "q": self.q, "c": self.field.c, "latency": latency,
            "ports": [(pt.name, pt.direction, pt.width) for pt in ports],
            "nodes": [(n.name, n.op, n.parents, n.kind, n.depth, n.width,
                       n.is_complex, n.const, n.dbg_offset) for n in nodes],
        }, sort_keys=True)
        return VerilogBundle(
            verilog_source=src, testbench_source=tb,
            module_name=self.module_name, ports=ports, latency=latency,
            field_p=self.p, field_q=self.q, num_vectors=nvec, vectors=lanes,
            nodes=nodes, debug=self.debug, dbg_width=dbg_width,
            dbg_vectors=dbg_vecs, program_hash=zlib.crc32(key.encode()))

    def _manifest(self, order) -> tuple[list[NodeInfo], int]:
        """Per-node hardware facts; with debug on, assigns dbg_state offsets."""
        alias = (IRMeasure, IRCompose, IRRegion, IRCatalyticRegion)
        nodes, off = [], 0
        for name, node in order:
            v = self._vals[name]
            if v.is_const:
                kind = "const"
            elif name in self._input_names:
                kind = "input"
            elif isinstance(node, alias):
                kind = "alias"
            else:
                kind = "reg"
            width = v.width if not v.is_const else \
                (2 * self.pw if v.cplx else self.pw)
            info = NodeInfo(
                name=name, op=type(node).__name__,
                parents=[par.name for par in node.parents], kind=kind,
                depth=0 if v.is_const else v.depth, width=width,
                is_complex=v.cplx,
                const=v.const if v.is_const else None)
            if self.debug and kind in ("input", "reg"):
                info.dbg_offset = off
                off += width
            nodes.append(info)
        return nodes, off

    def _render(self, ports: list[VerilogPort], assigns: list[str],
                latency: int, out_names: list[str],
                nodes: list[NodeInfo], dbg_width: int) -> str:
        f = self.field
        L = ["`timescale 1ns/1ps",
             "// Generated by cmtz verilog backend. Do not edit.",
             f"// Field: F_{self.p}" + (f"^2 (c={f.c})" if f.is_complex else "")
             + f", working mod {self.q}; latency {latency} clocks",
             f"module {self.module_name} (",
             "    input clk,", "    input in_valid,"]
        for pt in ports:
            rng = f"[{pt.width - 1}:0] " if pt.width > 1 else ""
            L.append(f"    {pt.direction} {rng}{pt.name},")
        if self.debug:
            L.append(f"    output [{dbg_width - 1}:0] dbg_state,")
        L += ["    output out_valid", ");", ""]
        L += list(self._funcs.values())
        if self._funcs:
            L.append("")
        L += self._decls + [""]
        L.append(f"reg [{latency - 1}:0] vpipe;" if latency > 1
                 else "reg vpipe;")
        L.append("initial vpipe = 0;")
        L.append("always @(posedge clk) vpipe <= "
                 + (f"{{vpipe[{latency - 2}:0], in_valid}};"
                    if latency > 1 else "in_valid;"))
        L.append(f"assign out_valid = vpipe[{latency - 1}];"
                 if latency > 1 else "assign out_valid = vpipe;")
        L.append("")
        if self.debug:
            # every registered node and input, LSB first; settles `depth`
            # clocks after the inputs stop changing
            parts = [self._vals[n.name].sig for n in nodes
                     if n.dbg_offset is not None]
            assigns = assigns + [
                "// debug readback: " + ", ".join(
                    f"{n.name}@{n.dbg_offset}" for n in nodes
                    if n.dbg_offset is not None),
                "assign dbg_state = {" + ", ".join(reversed(parts)) + "};"]
        L += self._body + [""] + assigns + ["", "endmodule", ""]
        return "\n".join(L)

    # ── Testbench ────────────────────────────────────────────────────

    def _vectors(self, prog: IRProgram) -> list[dict[str, Union[int, tuple]]]:
        p = self.p
        ins = [(n, isinstance(prog.get_node(n), IRComplexEmbed))
               for n in self._input_names]

        def space():
            axes = []
            for _, cplx in ins:
                axes += [range(p)] * (2 if cplx else 1)
            return axes

        axes = space()
        total = p ** len(axes)
        if total <= 512:
            flat = list(itertools.product(*axes))
        else:
            rng = random.Random(self.seed)
            flat = [tuple([0] * len(axes)), tuple([p - 1] * len(axes))]
            flat += [tuple(rng.randrange(p) for _ in axes)
                     for _ in range(self.num_vectors)]
        vecs = []
        for row in flat:
            it = iter(row)
            vecs.append({n: ((next(it), next(it)) if cplx else next(it))
                         for n, cplx in ins})
        return vecs

    def _reference(self, prog: IRProgram, vec: dict, out_names: list[str]):
        regs = evaluate_all(prog, self.field, vec)
        return [regs[n] for n in out_names]

    def _testbench(self, prog: IRProgram, ports: list[VerilogPort],
                   latency: int, out_names: list[str],
                   nodes: list[NodeInfo]):
        pw = self.pw
        vecs = self._vectors(prog)
        n = len(vecs)
        ins = [pt for pt in ports if pt.direction == "input"]
        outs = [pt for pt in ports if pt.direction == "output"]

        def pack(v, cplx):
            return (v[0] | (v[1] << pw)) if cplx else v

        def lanes(vals):
            out = []
            for v in vals:
                out += list(v) if isinstance(v, tuple) else [v]
            return out

        lane_vectors = []
        dbg_vectors = []
        dbg_nodes = [n for n in nodes if n.dbg_offset is not None]

        T = ["`timescale 1ns/1ps",
             "// Generated by cmtz verilog backend. Expected values come from",
             "// the Python reference backend (cmtz/backends/python_ref.py).",
             f"module {self.module_name}_tb;",
             "    reg clk = 1'b0;", "    reg in_valid = 1'b0;"]
        for pt in ins:
            T.append(f"    reg [{pt.width - 1}:0] {pt.name} = 0;")
        T.append("    wire out_valid;")
        for pt in outs:
            T.append(f"    wire [{pt.width - 1}:0] {pt.name};")
        for pt in ins:
            T.append(f"    reg [{pt.width - 1}:0] v_{pt.name} [0:{n - 1}];")
        for pt in outs:
            T.append(f"    reg [{pt.width - 1}:0] e_{pt.name} [0:{n - 1}];")
        conns = ["        .clk(clk)", "        .in_valid(in_valid)"]
        conns += [f"        .{pt.name}({pt.name})" for pt in ins + outs]
        conns.append("        .out_valid(out_valid)")
        T.append(f"    {self.module_name} dut (\n" + ",\n".join(conns) + "\n    );")
        T += ["    always #5 clk = ~clk;",
              "    integer i, k, errors;", "    initial begin",
              "        errors = 0; k = 0;"]
        for i, vec in enumerate(vecs):
            regs = evaluate_all(prog, self.field, vec)
            ref = [regs[n] for n in out_names]
            if self.debug:
                dbg_vectors.append(lanes([regs[n.name] for n in dbg_nodes]))
            lane_vectors.append((lanes([vec[pt.node] for pt in ins]),
                                 lanes(ref)))
            for pt in ins:
                T.append(f"        v_{pt.name}[{i}] = "
                         f"{_lit(pt.width, pack(vec[pt.node], pt.is_complex))};")
            for pt, val in zip(outs, ref):
                T.append(f"        e_{pt.name}[{i}] = "
                         f"{_lit(pt.width, pack(val, pt.is_complex))};")
        T += ["        repeat (2) @(negedge clk);",
              f"        for (i = 0; i < {n}; i = i + 1) begin",
              "            in_valid = 1'b1;"]
        T += [f"            {pt.name} = v_{pt.name}[i];" for pt in ins]
        T += ["            @(negedge clk);", "        end",
              "        in_valid = 1'b0;",
              f"        repeat ({latency + 3}) @(negedge clk);",
              f"        if (errors == 0 && k == {n}) $display(\"PASS\");",
              f"        else $display(\"FAIL: %0d errors, %0d of {n} results\", "
              "errors, k);",
              "        $finish;", "    end",
              "    always @(posedge clk) if (out_valid) begin",
              f"        if (k >= {n}) errors = errors + 1;", "        else begin"]
        for pt in outs:
            T.append(f"            if ({pt.name} !== e_{pt.name}[k]) begin "
                     f"$display(\"FAIL: vector %0d {pt.name}=%0d expected %0d\", "
                     f"k, {pt.name}, e_{pt.name}[k]); errors = errors + 1; end")
        T += ["        end", "        k = k + 1;", "    end", "endmodule", ""]
        return "\n".join(T), n, lane_vectors, dbg_vectors


# ── Reference evaluation ─────────────────────────────────────────────────────

def evaluate_all(prog: IRProgram, ir_field: IRField,
                 overrides: Optional[dict] = None) -> dict:
    """Value of every node under the Python reference backend, with the named
    nodes (typically the embeds exposed as ports) forced to `overrides`."""
    overrides = overrides or {}
    ref = PythonBackend(ir_field)
    for name, node in prog.topo_order():
        if name in overrides:
            ref.registers[name] = overrides[name]
        elif isinstance(node, (IRRegion, IRCatalyticRegion)):
            # eval_catalytic would re-evaluate the inner nodes from their
            # literals and clobber the overridden inputs
            inner = node.inner_nodes[-1].name if node.inner_nodes else None
            ref.registers[name] = ref.registers.get(inner, 0)
        else:
            ref.registers[name] = ref._eval_node(node)
    return dict(ref.registers)


# ── Public API ───────────────────────────────────────────────────────────────

def emit_verilog(prog: IRProgram, module_name: str = "cmtz_top",
                 inputs: Union[list[str], str, None] = None,
                 num_vectors: int = 64, seed: int = 0,
                 debug: bool = False) -> VerilogBundle:
    """Emit a pipelined Verilog module and self-checking testbench.

    Args:
        prog: Lowered and analyzed IR program
        module_name: top module name (testbench is `<module_name>_tb`)
        inputs: embed/cembed node names to expose as ports; default is the
            leaf embeds feeding an IRMul; "all" exposes every leaf embed that
            an output depends on
        num_vectors: random test vectors when the input space exceeds 512
        seed: RNG seed for those vectors
        debug: add a `dbg_state` output with every input and registered node
            (see NodeInfo.dbg_offset); the UART wrapper reads it back

    Raises:
        ValueError for a program without a field, or with matpow / unsupported
        node combinations.
    """
    if prog.ir_field is None:
        raise ValueError("Program has no field annotation — cannot emit Verilog")
    return VerilogEmitter(prog.ir_field, module_name, inputs,
                          num_vectors, seed, debug).generate(prog)

"""cmtz-core backend: lower an IR DAG to a program for the reusable processor.

Where the Verilog backend builds a new circuit per program, this backend
emits instructions for one fixed core (`cmtz/isa/`), so a program loads in
milliseconds instead of being synthesised.

Design:
  - Same folding rule as the Verilog backend: anything built only from
    constants is evaluated at emit time by the Python reference backend, so
    only runtime values become instructions.
  - Inputs default to every leaf embed some output depends on (`inputs=`
    overrides); the host writes them to their registers before a run.
  - Outputs are the measure nodes (or the sinks, if there are none), read
    back from their registers after HALT.
  - Linear-scan register allocation from R1..R23; R24..R31 and X4..X15
    belong to the runtime library (`isa/lib/matrix.s`). A spill is an error.
  - matpow is executable: the matrix is a runtime value the host writes to
    data memory, and the digit/bit structure of Lemma 3.12 (as in
    `PythonBackend.eval_matpow_matrix`) is unrolled at compile time into
    calls to the library. The node's scalar value stays 0, as in the
    reference backend; the result matrix is an output slot in data memory.
  - `interrupt { }` (IRCatalyticRegion) becomes the interrupt handler. Main
    ends with WAITI; HALT so a host-triggered handler always runs once. The
    handler's registers are disjoint from every register main uses, and it
    may read only inputs and constants from main. `isa_catalytic` verifies
    the allocated handler restores its borrowed registers.
  - `catalytic { }` (IRRegion) is advisory and compiles inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ..field import IRField, primitive_roots_Fp, to_base
from ..ir_nodes import (
    IRProgram, IRNode, IREmbed, IRPrimitiveRoots, IRRotate, IRPoly, IRMatPow,
    IRMeasure, IRCompose, IRRegion, IRCatalyticRegion, IRCycloPhiPoly, IRAdd,
    IRComplexEmbed, IRConjugate, IRMagnitudeSq, IRMul,
)
from ..analysis.isa_catalytic import verify_handler
from ..isa.asm import assemble
from ..isa.image import CoreConfig, MatrixSlot, ProgramImage
from .python_ref import PythonBackend

LIB_DIR = Path(__file__).resolve().parent.parent / "isa" / "lib"
MAIN_POOL = list(range(1, 24))      # R1..R23; R24..R31 are the library's


@dataclass
class IsaBundle:
    """Complete output from the cmtz-core backend."""
    image: ProgramImage
    core: CoreConfig
    asm: str                        # main + handler + library, assembled
    inputs: list[str]
    outputs: list[str]
    matrices_in: dict[str, int] = field(default_factory=dict)     # name -> n
    matrices_out: dict[str, str] = field(default_factory=dict)    # out -> M


@dataclass
class _Val:
    const: Optional[int] = None
    reg: str = ""               # virtual register name, if runtime

    @property
    def is_const(self) -> bool:
        return self.const is not None


class _Block:
    """Straight-line code over virtual registers for one context."""

    def __init__(self, name: str):
        self.name = name
        self.code: list[tuple[str, list]] = []   # (op, operands); vregs as str

    def emit(self, op: str, *ops) -> None:
        self.code.append((op, list(ops)))


class IsaEmitter:
    def __init__(self, ir_field: IRField, core: Optional[CoreConfig] = None,
                 inputs: Union[list[str], str, None] = None,
                 matrix_dims: Optional[dict[str, int]] = None):
        if ir_field.is_complex:
            raise ValueError("cmtz-core v1 is real-field only; complex "
                             "(F_{p^2}) programs are not supported yet")
        self.field = ir_field
        self.p, self.q = ir_field.p, ir_field.q
        self.core = core or CoreConfig(self.p, self.q)
        if (self.core.p, self.core.q) != (self.p, self.q):
            raise ValueError(f"program field ({self.p}, {self.q}) does not "
                             f"match the core ({self.core.p}, {self.core.q})")
        self.requested_inputs = inputs
        self.matrix_dims = matrix_dims or {}
        self.roots = primitive_roots_Fp(self.p)
        self._ref = PythonBackend(ir_field)
        self._vals: dict[str, _Val] = {}
        self._nv = 0
        self._dmem_next = 0
        self._dmem_free: dict[int, list[int]] = {}    # size -> addrs
        self.matrices_in: dict[str, MatrixSlot] = {}
        self.matrices_out: dict[str, MatrixSlot] = {}

    # ── selection ────────────────────────────────────────────────────

    def _outputs(self, prog: IRProgram) -> list[str]:
        measures = [n for n, node in prog.nodes() if isinstance(node, IRMeasure)]
        if measures:
            return measures
        used = {par.name for _, node in prog.nodes() for par in node.parents}
        skip = (IRPrimitiveRoots, IRRegion, IRCatalyticRegion, IRCycloPhiPoly)
        return [n for n, node in prog.nodes()
                if n not in used and not isinstance(node, skip)]

    def _inputs(self, prog: IRProgram, outs: list[str]) -> list[str]:
        leaves = [n for n, node in prog.nodes() if isinstance(node, IREmbed)]
        if self.requested_inputs not in (None, "all"):
            for n in self.requested_inputs:
                if n not in leaves:
                    raise ValueError(f"input '{n}' is not an embed node")
            return list(self.requested_inputs)
        anc: set[str] = set()

        def walk(node: IRNode) -> None:
            for par in node.parents:
                if par.name not in anc:
                    anc.add(par.name)
                    walk(par)
        for o in outs:
            anc.add(o)
            walk(prog.get_node(o))
        return [n for n in leaves if n in anc]

    # ── helpers ──────────────────────────────────────────────────────

    def _vreg(self) -> str:
        self._nv += 1
        return f"v{self._nv}"

    def _fold(self, node: IRNode, parents: list[_Val]) -> _Val:
        for par, v in zip(node.parents, parents):
            self._ref.registers[par.name] = v.const
        val = self._ref._eval_node(node)
        self._ref.registers[node.name] = val
        return _Val(const=val)

    def _in_reg(self, blk: _Block, v: _Val) -> str:
        """A vreg holding `v` (loads a constant with LDI)."""
        if not v.is_const:
            return v.reg
        r = self._vreg()
        blk.emit("LDI", r, v.const)
        return r

    def _dalloc(self, size: int) -> int:
        free = self._dmem_free.get(size)
        if free:
            return free.pop()
        addr = self._dmem_next
        self._dmem_next += size
        if self._dmem_next > self.core.dmem_size:
            raise ValueError("matrices do not fit in data memory "
                             f"({self.core.dmem_size} words)")
        return addr

    def _dfree(self, addr: int, size: int) -> None:
        self._dmem_free.setdefault(size, []).append(addr)

    # ── node lowering ────────────────────────────────────────────────

    def _lower(self, node: IRNode, blk: _Block) -> _Val:
        parents = [self._vals[p.name] for p in node.parents]

        if isinstance(node, (IRComplexEmbed, IRConjugate, IRMagnitudeSq)):
            raise ValueError(f"{type(node).__name__} '{node.name}': complex "
                             "values are not supported by cmtz-core v1")
        if isinstance(node, (IRRegion, IRCatalyticRegion)):
            last = node.inner_nodes[-1].name if node.inner_nodes else None
            return self._vals.get(last, _Val(const=0))
        if isinstance(node, (IRPrimitiveRoots, IRCycloPhiPoly)):
            return _Val(const=0)
        if isinstance(node, IREmbed):
            if node.name in self.input_names:
                return _Val(reg=self._input_vreg[node.name])
            return self._fold(node, parents)
        if isinstance(node, (IRCompose, IRMeasure)):
            # measure modes are the identity on values already in [0, p)
            if not parents:
                return _Val(const=0)
            return parents[-1] if isinstance(node, IRCompose) else parents[0]
        if isinstance(node, IRMatPow):
            self._lower_matpow(node, blk)
            return _Val(const=0)        # scalar value, as in the reference
        if all(v.is_const for v in parents):
            return self._fold(node, parents)

        out = self._vreg()
        p = self.p
        if isinstance(node, IRRotate):
            w = self.roots[node.j % len(self.roots)]
            blk.emit("MOV" if w == 1 else "MULI", out, parents[0].reg,
                     *(() if w == 1 else (w,)))
        elif isinstance(node, IRPoly):
            # the accumulator holds values in [0, q); only `out` is in [0, p)
            coeffs = [c % self.q for c in node.coeffs] or [0]
            x, acc = parents[0].reg, self._vreg()
            blk.emit("LDI", acc, coeffs[-1])
            for c in reversed(coeffs[:-1]):
                blk.emit("HORNER", acc, x, c)
            blk.emit("REDP", out, acc)
        elif isinstance(node, IRAdd):
            a, b = parents
            blk.emit("ADD", out, self._in_reg(blk, a), self._in_reg(blk, b))
        elif isinstance(node, IRMul):
            a, b = parents
            if a.is_const or b.is_const:
                c, r = (a, b) if a.is_const else (b, a)
                blk.emit("MULI", out, r.reg, c.const % p)
            else:
                blk.emit("MUL", out, a.reg, b.reg)
        else:
            raise ValueError(f"{type(node).__name__} '{node.name}' unsupported")
        return _Val(reg=out)

    def _lower_matpow(self, node: IRMatPow, blk: _Block) -> None:
        """Unroll Lemma 3.12 into library calls (see eval_matpow_matrix)."""
        mname = node.matrix_name
        n = self.matrix_dims.get(mname) or node.n_dim
        if not n:
            raise ValueError(f"matpow '{node.name}': give the dimension of "
                             f"matrix '{mname}' with matrix_dims=")
        size = n * n
        if mname not in self.matrices_in:
            self.matrices_in[mname] = MatrixSlot(self._dalloc(size), n)
        src = self.matrices_in[mname]
        if src.n != n:
            raise ValueError(f"matrix '{mname}' used with two dimensions")
        owned: set[int] = set()      # buffers this matpow may free

        def call(fn: str, a: int = 0, b: int = 0, c: int = 0) -> None:
            blk.emit("LIX", "X4", a)
            blk.emit("LIX", "X5", b)
            blk.emit("LIX", "X6", c)
            blk.emit("LIX", "X7", n)
            blk.emit("CALL", fn)

        def new() -> int:
            addr = self._dalloc(size)
            owned.add(addr)
            return addr

        def release(addr: int) -> None:
            if addr in owned:
                owned.discard(addr)
                self._dfree(addr, size)

        def ident() -> int:
            out = new()
            call("__matident", c=out)
            return out

        def mul(a: int, b: int) -> int:
            out = new()
            call("__matmul", a, b, out)
            return out

        def pow_small(m: int, k: int) -> int:
            # _mat_pow_small: k == 1 is a copy (an alias here, never written)
            if k == 0:
                return ident()
            if k == 1:
                return m
            result, base = None, m
            while k:
                if k & 1:
                    if result is None:      # I * base == base
                        result = base
                    else:
                        t = mul(result, base)
                        if result not in (m, base):
                            release(result)
                        result = t
                k >>= 1
                if k:                       # the last squaring is dead
                    t = mul(base, base)
                    if base not in (m, result):
                        release(base)
                    base = t
            if base not in (m, result):
                release(base)
            return result

        digits = to_base(node.d, node.delta)
        result: Optional[int] = None        # None = identity
        mk = src.addr
        for i, alpha in enumerate(digits):
            if alpha > 0:
                block = pow_small(mk, alpha)
                if result is None:
                    result = block
                else:
                    t = mul(result, block)
                    if result != mk:
                        release(result)
                    if block != mk:
                        release(block)
                    result = t
            if i + 1 < len(digits):         # the last advance is dead
                t = pow_small(mk, node.delta)
                if mk not in (src.addr, result):
                    release(mk)
                mk = t
        out = self._dalloc(size)
        if result is None:
            call("__matident", c=out)
        else:
            call("__matcopy", a=result, c=out)
        for b in list(owned):
            release(b)
        self.matrices_out[node.name] = MatrixSlot(out, n)

    # ── allocation ───────────────────────────────────────────────────

    def _allocate(self, blk: _Block, pool: list[int], pinned: dict[str, int],
                  live_out: set[str],
                  shared: Optional[dict[str, int]] = None) -> dict[str, int]:
        """Linear scan over straight-line code. `pinned` vregs already have
        registers from `pool` and stay live for the whole block; `shared`
        vregs live in registers outside `pool` (read only here); `live_out`
        vregs stay live to the end."""
        last: dict[str, int] = {}
        for i, (_, ops) in enumerate(blk.code):
            for o in ops:
                if isinstance(o, str) and o.startswith("v"):
                    last[o] = i
        for v in live_out:
            last[v] = len(blk.code)
        regs = {**(shared or {}), **pinned}
        keep = set(regs)
        free = [r for r in pool if r not in pinned.values()]
        for i, (_, ops) in enumerate(blk.code):
            # operands whose last use is here are freed before the
            # destination is allocated, so `ADD R1, R1, R2` style reuse works
            for o in ops[1:]:
                if (isinstance(o, str) and o.startswith("v") and o not in keep
                        and last.get(o) == i and regs.get(o) not in free):
                    free.insert(0, regs[o])
            dst = ops[0] if ops else None
            if isinstance(dst, str) and dst.startswith("v") and dst not in regs:
                if not free:
                    raise ValueError(f"{blk.name}: out of registers "
                                     f"({len(pool)} available, spills are "
                                     "not supported)")
                regs[dst] = free.pop(0)
                if last.get(dst) == i:          # dead value
                    free.insert(0, regs[dst])
        return regs

    @staticmethod
    def _render(blk: _Block, regs: dict[str, int]) -> list[str]:
        out = []
        for op, ops in blk.code:
            parts = []
            for o in ops:
                if isinstance(o, str) and o.startswith("v"):
                    parts.append(f"R{regs[o]}")
                else:
                    parts.append(str(o))
            out.append(f"        {op:<6} {', '.join(parts)}".rstrip())
        return out

    # ── driver ───────────────────────────────────────────────────────

    def generate(self, prog: IRProgram) -> IsaBundle:
        outs = self._outputs(prog)
        self.input_names = self._inputs(prog, outs)
        self._input_vreg = {n: self._vreg() for n in self.input_names}

        regions = [node for _, node in prog.nodes()
                   if isinstance(node, IRCatalyticRegion)]
        if len(regions) > 1:
            raise ValueError("cmtz-core v1 supports one interrupt handler")
        h_names = {n.name for r in regions for n in r.inner_nodes}

        main, handler = _Block("main"), _Block("handler")
        for name, node in prog.topo_order():
            in_h = name in h_names
            if not in_h and any(par.name in h_names for par in node.parents) \
                    and not isinstance(node, IRMeasure):
                raise ValueError(f"'{name}': main code may not depend on "
                                 "interrupt handler results (only measures)")
            self._vals[name] = self._lower(node, handler if in_h else main)

        # constant outputs still get a register, so the host reads every
        # output the same way
        for o in outs:
            if self._vals[o].is_const:
                r = self._vreg()
                main.emit("LDI", r, self._vals[o].const)
                self._vals[o] = _Val(reg=r)

        def vregs(blk: _Block, reads_only: bool = False) -> set[str]:
            return {o for _, ops in blk.code
                    for o in (ops[1:] if reads_only else ops)
                    if isinstance(o, str) and o.startswith("v")}
        in_v = set(self._input_vreg.values())
        out_v = {self._vals[o].reg for o in outs}
        h_def = {ops[0] for _, ops in handler.code}
        if (vregs(handler, True) - h_def) - in_v:
            raise ValueError("the interrupt handler may read only inputs, "
                             "constants and its own values")

        # Inputs main reads (or outputs directly) are pinned in main's pool
        # for the whole run; inputs only the handler reads are pinned in the
        # handler's pool. Everything main touches is off limits to the
        # handler, since the interrupt can land between any two instructions.
        main_in = [v for v in self._input_vreg.values()
                   if v in vregs(main, True) or
                   (v in out_v and v not in vregs(handler, True))]
        h_in = [v for v in self._input_vreg.values()
                if v not in main_in and v in vregs(handler, True)]
        unused = [v for v in self._input_vreg.values()
                  if v not in main_in and v not in h_in]
        main_in += unused
        if len(in_v) > len(MAIN_POOL):
            raise ValueError("too many inputs for the register file")
        pinned = {v: MAIN_POOL[i] for i, v in enumerate(main_in)}
        main_regs = self._allocate(main, MAIN_POOL, pinned, out_v - h_def)
        used = set(main_regs.values())
        h_pool = [r for r in MAIN_POOL if r not in used]
        if len(h_in) > len(h_pool):
            raise ValueError("no registers left for the interrupt handler")
        h_pin = {v: h_pool[i] for i, v in enumerate(h_in)}
        h_regs = self._allocate(handler, h_pool, h_pin, out_v & h_def,
                                shared={v: r for v, r in main_regs.items()
                                        if v in in_v})
        regs = {**main_regs, **h_regs}
        lines = [".entry main"]
        if regions:
            lines.append(".handler __isr")
        lines.append("main:")
        lines += self._render(main, regs)
        if regions:
            lines.append("        WAITI")
        lines.append("        HALT")
        if regions:
            lines.append("__isr:")
            lines += self._render(handler, regs)
            lines.append("        RETI")
        text = "\n".join(lines) + "\n"
        lib = (LIB_DIR / "matrix.s").read_text() if self.matrices_out else ""
        asm = assemble(text, lib)

        borrowed: list[int] = []
        for r in regions:
            for name in r.catalytic_regs:
                if name not in self._vals:
                    raise ValueError(f"interrupt restores unknown value '{name}'")
                v = self._vals[name]
                if not v.is_const:
                    borrowed.append(regs[v.reg])
        image = ProgramImage(
            p=self.p, q=self.q, words=asm.words, entry=asm.entry,
            handler=asm.handler,
            inputs={n: regs[v] for n, v in self._input_vreg.items()},
            outputs={o: regs[self._vals[o].reg] for o in outs},
            matrices_in=dict(self.matrices_in),
            matrices_out=dict(self.matrices_out),
            borrowed=sorted(set(borrowed)),
            handler_scratch=sorted(set(h_pool)),
            listing=[str(i) for i in asm.instrs],
        )
        image.validate(self.core)
        verify_handler(image)
        return IsaBundle(image=image, core=self.core, asm=text + lib,
                         inputs=list(self.input_names), outputs=outs,
                         matrices_in={k: s.n for k, s in self.matrices_in.items()},
                         matrices_out={k: k for k in self.matrices_out})


def emit_isa(prog: IRProgram, core: Optional[CoreConfig] = None,
             inputs: Union[list[str], str, None] = None,
             matrix_dims: Optional[dict[str, int]] = None) -> IsaBundle:
    """Compile an IR program to a cmtz-core program image.

    Args:
        prog: lowered and analysed IR program (real field)
        core: the core to target; default CoreConfig(p, q) of the program
        inputs: embed names the host writes; default every leaf embed an
            output depends on
        matrix_dims: n for each matrix a matpow node reads, by name
    """
    if prog.ir_field is None:
        raise ValueError("Program has no field annotation")
    return IsaEmitter(prog.ir_field, core, inputs, matrix_dims).generate(prog)

"""WebAssembly Text Format (WAT) backend for Cook-Mertz.

Emits a fully unrolled WAT module implementing the Cook-Mertz pipeline.
Operations use 32/64-bit integer modular arithmetic — no floats anywhere.

Design:
  - Root table in linear memory at byte offset 0 (m × 4 bytes, little-endian i32)
  - Real registers: one i32 local per IR node
  - Complex registers: two i32 locals per node ($name_re, $name_im)
  - Measured values written to output area starting at byte offset m*4
  - Last measured value returned from the exported "compute" function
  - Multiply uses i64 widening: stays exact for any p < 2^31

Two-pass emission:
  1. Collect all local declarations (WAT requires locals before any instructions)
  2. Emit instruction body in topological order

Assembly and execution:
    wat2wasm output.wat -o output.wasm     # WABT toolchain
    wasm-opt -O3 output.wasm -o output.wasm  # optional Binaryen optimisation

Host setup (JavaScript / Node):
    const bytes = fs.readFileSync('output.wasm');
    const { instance } = await WebAssembly.instantiate(bytes);
    // Copy root table into WASM memory (byte offset 0):
    const view = new DataView(instance.exports.memory.buffer);
    root_table.forEach((v, i) => view.setUint32(i * 4, v, true));
    const result = instance.exports.compute();
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from ..ir_nodes import (
    IRProgram, IRNode, IREmbed, IRPrimitiveRoots, IRRotate,
    IRPoly, IRMatPow, IRMeasure, IRCompose, IRRegion, IRCatalyticRegion,
    IRCycloPhiPoly, IRAdd, IRComplexEmbed, IRConjugate, IRMagnitudeSq,
)
from ..field import IRField, primitive_roots_Fp, to_base


# ── Output types ─────────────────────────────────────────────────────────────

@dataclass
class MemoryLayout:
    """Describes the linear memory layout of the emitted module."""
    root_table_offset: int   # always 0
    root_table_bytes: int    # m * 4
    output_offset: int       # root_table_bytes; one i32 per measure
    measure_count: int
    total_bytes: int


@dataclass
class WasmBundle:
    """Complete output from the WASM backend."""
    wat_source: str
    root_table: list[int]
    layout: MemoryLayout
    field_p: int
    field_q: int

    def assemble_command(self, input_path: str = "out.wat",
                         output_path: str = "out.wasm") -> str:
        """Return the wat2wasm command to assemble this module."""
        return f"wat2wasm {input_path} -o {output_path}"

    def node_command(self, input_path: str = "out.wasm",
                     output_path: str = "opt.wasm") -> str:
        """Return an optional wasm-opt optimisation command."""
        return f"wasm-opt -O3 {input_path} -o {output_path}"


# ── WAT emitter ───────────────────────────────────────────────────────────────

class WatEmitter:
    """Walk the IR DAG and emit a WAT module, fully unrolled."""

    def __init__(self, ir_field: IRField):
        self.field = ir_field
        self.p = ir_field.p
        self.q = ir_field.q
        self.m = ir_field.m
        self.roots = list(primitive_roots_Fp(ir_field.p))

        # Emitted lines for different sections
        self._helper_lines: list[str] = []
        self._decl_lines: list[str] = []    # (local $name i32) declarations
        self._body_lines: list[str] = []    # instruction body

        # IR node name → WAT local name(s); complex nodes map to (re, im)
        self._var_map: dict[str, str | tuple[str, str]] = {}

        # Measure tracking
        self._measure_count = 0
        self._last_measure_var: Optional[str] = None

    # ── Root table encoding ───────────────────────────────────────────

    def _encode_root_table(self) -> str:
        """Encode root table as a WAT string literal (\\xx hex escapes)."""
        raw = b''.join(struct.pack('<I', r) for r in self.roots)
        return ''.join(f'\\{b:02x}' for b in raw)

    # ── Helper functions ──────────────────────────────────────────────

    def _emit_helpers(self) -> None:
        p, q, c = self.p, self.q, self.field.c
        h = self._helper_lines

        def line(s: str) -> None:
            h.append(s)

        # mul_mod_p: 64-bit widening multiply, reduce mod p
        line(f'  (func $mul_p (param $a i32) (param $b i32) (result i32)')
        line(f'    (i32.wrap_i64')
        line(f'      (i64.rem_u')
        line(f'        (i64.mul (i64.extend_i32_u (local.get $a))')
        line(f'                 (i64.extend_i32_u (local.get $b)))')
        line(f'        (i64.const {p}))))')

        # mul_mod_q: same but mod q (for polynomial Horner steps)
        line(f'  (func $mul_q (param $a i32) (param $b i32) (result i32)')
        line(f'    (i32.wrap_i64')
        line(f'      (i64.rem_u')
        line(f'        (i64.mul (i64.extend_i32_u (local.get $a))')
        line(f'                 (i64.extend_i32_u (local.get $b)))')
        line(f'        (i64.const {q}))))')

        # add_mod_p
        line(f'  (func $add_p (param $a i32) (param $b i32) (result i32)')
        line(f'    (i32.rem_u (i32.add (local.get $a) (local.get $b))')
        line(f'               (i32.const {p})))')

        if self.field.is_complex:
            # magsq_fp2: |a+bi|² = a² - b²·c mod p
            # Computed as (a²%p + p - b²·c%p) % p to avoid unsigned underflow
            line(f'  ;; |a+bi|² = a² - b²·c mod p  (c = non-residue = {c})')
            line(f'  (func $magsq (param $re i32) (param $im i32) (result i32)')
            line(f'    (i32.rem_u')
            line(f'      (i32.add')
            line(f'        (call $mul_p (local.get $re) (local.get $re))')
            line(f'        (i32.sub (i32.const {p})')
            line(f'          (call $mul_p')
            line(f'            (call $mul_p (local.get $im) (local.get $im))')
            line(f'            (i32.const {c}))))')
            line(f'      (i32.const {p})))')

    # ── Local collection (pass 1) ─────────────────────────────────────

    def _collect_locals(self, prog: IRProgram) -> None:
        """Walk the IR and emit (local $name i32) declarations."""
        for name, node in prog.topo_order():
            if self._is_complex(node):
                re_var = f'${name}_re'
                im_var = f'${name}_im'
                self._var_map[name] = (re_var, im_var)
                self._decl_lines.append(f'    (local {re_var} i32)')
                self._decl_lines.append(f'    (local {im_var} i32)')
            else:
                var = f'${name}'
                self._var_map[name] = var
                self._decl_lines.append(f'    (local {var} i32)')

    def _is_complex(self, node: IRNode) -> bool:
        """True iff this node produces a complex (F_{p²}) value."""
        return isinstance(node, (IRComplexEmbed, IRConjugate)) or (
            isinstance(node, IRAdd) and
            self.field.is_complex and
            any(isinstance(p, (IRComplexEmbed, IRConjugate))
                for p in node.parents)
        )

    # ── Instruction body emission (pass 2) ───────────────────────────

    def _emit(self, line: str) -> None:
        self._body_lines.append(line)

    def _emit_body(self, prog: IRProgram) -> None:
        for name, node in prog.topo_order():
            self._emit_node(name, node)

    def _emit_node(self, name: str, node: IRNode) -> None:
        if isinstance(node, IREmbed):
            self._emit_embed(name, node)
        elif isinstance(node, IRPrimitiveRoots):
            self._emit_primitive_roots(name, node)
        elif isinstance(node, IRRotate):
            self._emit_rotate(name, node)
        elif isinstance(node, IRPoly):
            self._emit_poly(name, node)
        elif isinstance(node, IRMeasure):
            self._emit_measure(name, node)
        elif isinstance(node, IRCompose):
            self._emit_compose(name, node)
        elif isinstance(node, (IRRegion, IRCatalyticRegion)):
            self._emit_region(name, node)
        elif isinstance(node, IRMatPow):
            self._emit_matpow_stub(name, node)
        elif isinstance(node, IRAdd):
            self._emit_add(name, node)
        elif isinstance(node, IRComplexEmbed):
            self._emit_complex_embed(name, node)
        elif isinstance(node, IRConjugate):
            self._emit_conjugate(name, node)
        elif isinstance(node, IRMagnitudeSq):
            self._emit_magsq(name, node)
        elif isinstance(node, IRCycloPhiPoly):
            self._emit_cyclo_phi(name, node)

    def _emit_embed(self, name: str, node: IREmbed) -> None:
        var = self._var_map[name]
        j = node.psi % self.m
        if j == 0:
            self._emit(f'    ;; embed({node.index}, 0) = omega^0 = 1')
            self._emit(f'    (local.set {var} (i32.const 1))')
        else:
            byte_off = j * 4
            val = self.roots[j]
            self._emit(f'    ;; embed({node.index}, {node.psi}) = omega^{j} = {val}')
            self._emit(f'    (local.set {var} (i32.load (i32.const {byte_off})))')

    def _emit_primitive_roots(self, name: str, node: IRPrimitiveRoots) -> None:
        var = self._var_map[name]
        self._emit(f'    ;; roots({node.ir_field.p if node.ir_field else "?"}'
                   f') — table in memory at offset 0')
        self._emit(f'    (local.set {var} (i32.const 0))')

    def _emit_rotate(self, name: str, node: IRRotate) -> None:
        var = self._var_map[name]
        src_var = self._parent_var(node, 0)
        j = node.j % self.m
        byte_off = j * 4
        omega_val = self.roots[j]
        self._emit(f'    ;; rotate by omega^{j} = {omega_val}')
        self._emit(f'    (local.set {var}')
        self._emit(f'      (call $mul_p {src_var}')
        self._emit(f'        (i32.load (i32.const {byte_off}))))')

    def _emit_poly(self, name: str, node: IRPoly) -> None:
        """Horner evaluation in F_q, then reduce mod p."""
        var = self._var_map[name]
        x_var = self._parent_var(node, 0)
        coeffs_desc = list(reversed(node.coeffs))
        if not coeffs_desc:
            self._emit(f'    (local.set {var} (i32.const 0))')
            return

        self._emit(f'    ;; poly degree {node.degree}, Horner in F_{self.q}')
        self._emit(f'    (local.set {var} (i32.const {coeffs_desc[0]}))')
        for i, c in enumerate(coeffs_desc[1:], 1):
            self._emit(f'    (local.set {var}  ;; step {i}/{len(coeffs_desc)-1}')
            self._emit(f'      (i32.rem_u')
            self._emit(f'        (i32.add (call $mul_q {var} {x_var})')
            self._emit(f'                 (i32.const {c}))')
            self._emit(f'        (i32.const {self.q})))')
        self._emit(f'    (local.set {var} (i32.rem_u {var} (i32.const {self.p})))')

    def _emit_measure(self, name: str, node: IRMeasure) -> None:
        var = self._var_map[name]
        src = self._parent_var(node, 0)
        out_byte = self.m * 4 + self._measure_count * 4

        self._emit(f'    ;; measure({node.src.name if node.src else "?"}, {node.mode})')
        if node.mode == 'mod_p':
            self._emit(f'    (local.set {var} (i32.rem_u {src} (i32.const {self.p})))')
        else:
            self._emit(f'    (local.set {var} {src})')

        # Write to output area in memory
        self._emit(f'    (i32.store (i32.const {out_byte}) (local.get {var}))')

        self._last_measure_var = var
        self._measure_count += 1

    def _emit_compose(self, name: str, node: IRCompose) -> None:
        var = self._var_map[name]
        # Composition result = last parent's value (topo order guarantees evaluation)
        if node.parents:
            last = self._var_map.get(node.parents[-1].name, '(i32.const 0)')
            last_str = last if isinstance(last, str) else last[0]
            self._emit(f'    ;; compose -> alias last parent')
            self._emit(f'    (local.set {var} (local.get {last_str}))')
        else:
            self._emit(f'    (local.set {var} (i32.const 0))')

    def _emit_region(self, name: str,
                     node: IRRegion | IRCatalyticRegion) -> None:
        kind = ('interrupt' if isinstance(node, IRCatalyticRegion)
                else 'catalytic')
        var = self._var_map[name]
        self._emit(f'    ;; {kind} region "{name}" restoring {node.catalytic_regs}')
        # Inner nodes were already emitted during topo traversal of the outer program.
        # The region node itself just aliases its last inner node's value.
        if node.inner_nodes:
            last_inner = node.inner_nodes[-1]
            inner_var = self._var_map.get(last_inner.name, '(i32.const 0)')
            inner_str = inner_var if isinstance(inner_var, str) else inner_var[0]
            self._emit(f'    (local.set {var} (local.get {inner_str}))')
        else:
            self._emit(f'    (local.set {var} (i32.const 0))')

    def _emit_matpow_stub(self, name: str, node: IRMatPow) -> None:
        var = self._var_map[name]
        digits = to_base(node.d, node.delta)
        self._emit(f'    ;; === matpow stub: {node.matrix_name}^{node.d} ===')
        self._emit(f'    ;; delta={node.delta} epsilon={node.epsilon} '
                   f'{len(digits)} base-delta digits')
        self._emit(f'    ;; full impl requires tiled matmul dispatch (separate module)')
        self._emit(f'    (local.set {var} (i32.const 0))  ;; placeholder')

    def _emit_add(self, name: str, node: IRAdd) -> None:
        if self._is_complex(node):
            re_var, im_var = self._var_map[name]   # type: ignore[misc]
            a = self._parent_var_complex(node, 0)
            b = self._parent_var_complex(node, 1)
            self._emit(f'    ;; add (complex) -> {name}')
            self._emit(f'    (local.set {re_var} (call $add_p {a[0]} {b[0]}))')
            self._emit(f'    (local.set {im_var} (call $add_p {a[1]} {b[1]}))')
        else:
            var = self._var_map[name]
            a = self._parent_var(node, 0)
            b = self._parent_var(node, 1)
            self._emit(f'    ;; add (real) -> {name}')
            self._emit(f'    (local.set {var} (call $add_p {a} {b}))')

    def _emit_complex_embed(self, name: str, node: IRComplexEmbed) -> None:
        re_var, im_var = self._var_map[name]   # type: ignore[misc]
        re_j = node.re_psi % self.m
        im_j = node.im_psi % self.m
        re_val = self.roots[re_j]
        im_val = self.roots[im_j]
        self._emit(f'    ;; cembed({node.re_psi}, {node.im_psi}) = '
                   f'{re_val}+{im_val}i')
        if re_j == 0:
            self._emit(f'    (local.set {re_var} (i32.const 1))')
        else:
            self._emit(f'    (local.set {re_var} (i32.load (i32.const {re_j * 4})))')
        if im_j == 0:
            self._emit(f'    (local.set {im_var} (i32.const 1))')
        else:
            self._emit(f'    (local.set {im_var} (i32.load (i32.const {im_j * 4})))')

    def _emit_conjugate(self, name: str, node: IRConjugate) -> None:
        re_var, im_var = self._var_map[name]   # type: ignore[misc]
        src = self._parent_var_complex(node, 0)
        # src[0] and src[1] are already "(local.get $name)" expressions
        self._emit(f'    ;; conj({node.parents[0].name}) -> {name}: re same, im negated')
        self._emit(f'    (local.set {re_var} {src[0]})')
        # (-im) mod p = (p - im) % p; safe since im ∈ [0, p)
        self._emit(f'    (local.set {im_var}')
        self._emit(f'      (i32.rem_u')
        self._emit(f'        (i32.sub (i32.const {self.p}) {src[1]})')
        self._emit(f'        (i32.const {self.p})))')

    def _emit_magsq(self, name: str, node: IRMagnitudeSq) -> None:
        var = self._var_map[name]
        src = self._parent_var_complex(node, 0)
        self._emit(f'    ;; magsq({node.parents[0].name}) = |z|² mod {self.p}')
        self._emit(f'    (local.set {var} (call $magsq {src[0]} {src[1]}))')

    def _emit_cyclo_phi(self, name: str, node: IRCycloPhiPoly) -> None:
        var = self._var_map[name]
        self._emit(f'    ;; cyclo_phi({node.n}): {len(node.poly_coeffs)} coefficients')
        self._emit(f'    ;; (FIR filter use — coefficients in memory would require '
                   f'separate data segment)')
        self._emit(f'    (local.set {var} (i32.const 0))')

    # ── Parent resolution helpers ─────────────────────────────────────

    def _parent_var(self, node: IRNode, idx: int) -> str:
        """Get the WAT local ref for the idx-th parent (real register)."""
        if idx < len(node.parents):
            mapping = self._var_map.get(node.parents[idx].name)
            if isinstance(mapping, tuple):
                return f'(local.get {mapping[0]})'  # take re part if complex
            if mapping:
                return f'(local.get {mapping})'
        return '(i32.const 0)'

    def _parent_var_complex(self, node: IRNode,
                             idx: int) -> tuple[str, str]:
        """Get (re_ref, im_ref) WAT strings for the idx-th parent."""
        if idx < len(node.parents):
            mapping = self._var_map.get(node.parents[idx].name)
            if isinstance(mapping, tuple):
                return (f'(local.get {mapping[0]})',
                        f'(local.get {mapping[1]})')
            if mapping:
                return (f'(local.get {mapping})',
                        '(i32.const 0)')
        return ('(i32.const 0)', '(i32.const 0)')

    # ── Top-level assembly ────────────────────────────────────────────

    def generate(self, prog: IRProgram) -> WasmBundle:
        """Generate a complete WAT module from the IR program."""
        self._emit_helpers()
        self._collect_locals(prog)
        self._emit_body(prog)

        # Memory layout
        root_bytes = self.m * 4
        output_offset = root_bytes
        total_bytes = output_offset + self._measure_count * 4

        encoded = self._encode_root_table()

        # Return variable: last measured value, or 0 if no measures
        ret_var = (f'(local.get {self._last_measure_var})'
                   if self._last_measure_var else '(i32.const 0)')

        # Assemble sections
        lines: list[str] = []
        lines.append('(module')
        lines.append(f'  ;; Cook-Mertz WAT  |  F_{self.p}, working mod {self.q}'
                     f'  |  m={self.m}')
        lines.append(f'  ;; Root table: {self.m} × i32 at byte 0')
        lines.append(f'  ;; Output area: {self._measure_count} × i32 '
                     f'at byte {output_offset}')
        lines.append('')
        lines.append('  (memory 1)')
        lines.append(f'  (data (i32.const 0) "{encoded}")')
        lines.append('  (export "memory" (memory 0))')
        lines.append('')
        lines.extend(self._helper_lines)
        lines.append('')
        lines.append('  (func $compute (export "compute") (result i32)')

        if self._decl_lines:
            lines.extend(self._decl_lines)
            lines.append('')

        if self._body_lines:
            lines.extend(self._body_lines)
            lines.append('')

        lines.append(f'    {ret_var}')
        lines.append('  )')
        lines.append(')')

        wat = '\n'.join(lines)
        layout = MemoryLayout(
            root_table_offset=0,
            root_table_bytes=root_bytes,
            output_offset=output_offset,
            measure_count=self._measure_count,
            total_bytes=total_bytes,
        )
        return WasmBundle(
            wat_source=wat,
            root_table=self.roots,
            layout=layout,
            field_p=self.p,
            field_q=self.q,
        )


# ── Public API ───────────────────────────────────────────────────────────────

def emit_wasm(prog: IRProgram) -> WasmBundle:
    """Emit a WAT module from a compiled IR program.

    Args:
        prog: Lowered and analyzed IR program (must have ir_field set)

    Returns:
        WasmBundle with WAT source, root table, and memory layout

    Raises:
        ValueError if program has no field annotation
    """
    if prog.ir_field is None:
        raise ValueError(
            "Program has no field annotation — cannot emit WAT"
        )
    emitter = WatEmitter(prog.ir_field)
    return emitter.generate(prog)

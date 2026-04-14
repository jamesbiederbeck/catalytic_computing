"""GLSL compute shader backend for Cook-Mertz v2.

Emits a fully unrolled GLSL 4.50 compute shader implementing the
Cook-Mertz pipeline. All operations are branch-free integer modular
arithmetic, matching the IR's algebraic semantics exactly.

Design constraints:
  - Root-of-unity table loaded as a SSBO (precomputed on CPU)
  - All mod operations use explicit `%` on uint (GLSL 4.50+)
  - No dynamic loops or conditionals — fully unrolled from the IR DAG
  - Workgroup size set at compile time from the IR program structure
  - Polynomial evaluation via Horner's method, one multiply-add per line

The output is a GLSLShaderBundle containing:
  - The GLSL source string
  - The root table as a flat uint array (for SSBO upload)
  - Buffer layout metadata (binding points, sizes)

For SPIR-V compilation, pipe the GLSL through glslangValidator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..ir_nodes import (
    IRProgram, IRNode, IREmbed, IRPrimitiveRoots, IRRotate,
    IRPoly, IRMatPow, IRMeasure, IRCompose, IRCatalyticRegion,
    IRCycloPhiPoly,
)
from ..field import IRField, primitive_roots_Fp


# ── Output types ─────────────────────────────────────────────────────────────

@dataclass
class BufferBinding:
    """Describes a SSBO or UBO binding in the emitted shader."""
    binding: int
    name: str
    element_type: str       # 'uint' or 'int'
    element_count: int
    usage: str              # 'readonly' | 'writeonly' | 'readwrite'


@dataclass
class GLSLShaderBundle:
    """Complete output from the GLSL backend."""
    glsl_source: str
    root_table: list[int]           # uint values for SSBO upload
    buffers: list[BufferBinding]
    workgroup_size: tuple[int, int, int]
    field_p: int
    field_q: int

    def spirv_compile_command(self, output_path: str = "shader.spv") -> str:
        """Return the glslangValidator command to compile to SPIR-V."""
        return f"glslangValidator -V --target-env vulkan1.2 -o {output_path} shader.comp"


# ── Code emitter ─────────────────────────────────────────────────────────────

class GLSLEmitter:
    """Walks the IR DAG and emits GLSL compute shader source."""

    def __init__(self, ir_field: IRField, workgroup_x: int = 64):
        self.field = ir_field
        self.p = ir_field.p
        self.q = ir_field.q
        self.m = ir_field.m
        self.workgroup_x = workgroup_x

        self.roots = list(primitive_roots_Fp(ir_field.p))
        self._lines: list[str] = []
        self._indent = 0
        self._var_counter = 0
        self._var_map: dict[str, str] = {}  # IR node name -> GLSL var name
        self._buffers: list[BufferBinding] = []
        self._next_binding = 0

    def emit(self, line: str) -> None:
        self._lines.append("    " * self._indent + line)

    def fresh_var(self, prefix: str = "v") -> str:
        self._var_counter += 1
        return f"{prefix}_{self._var_counter}"

    def alloc_binding(self, name: str, element_type: str,
                      count: int, usage: str) -> int:
        binding = self._next_binding
        self._next_binding += 1
        self._buffers.append(BufferBinding(
            binding=binding, name=name,
            element_type=element_type,
            element_count=count, usage=usage,
        ))
        return binding

    # ── Top-level shader generation ──────────────────────────────────

    def generate(self, prog: IRProgram) -> GLSLShaderBundle:
        """Generate a complete GLSL compute shader from the IR program."""

        # Header
        self.emit("#version 450")
        self.emit("")
        self.emit(f"// Cook-Mertz v2 compute shader")
        self.emit(f"// Field: F_{self.p}, working mod {self.q}")
        self.emit(f"// Root order m = {self.m}")
        self.emit("")

        # Workgroup layout
        self.emit(f"layout(local_size_x = {self.workgroup_x}, "
                  f"local_size_y = 1, local_size_z = 1) in;")
        self.emit("")

        # Root table SSBO
        root_binding = self.alloc_binding(
            "roots", "uint", self.m, "readonly"
        )
        self.emit(f"layout(std430, binding = {root_binding}) "
                  f"readonly buffer RootTable {{")
        self.emit(f"    uint omega[{self.m}];  "
                  f"// omega[j] = g^j mod {self.p}")
        self.emit("};")
        self.emit("")

        # Input SSBO
        # Size is determined by the program; default to m elements
        input_size = max(self.m, 1)
        input_binding = self.alloc_binding(
            "input_data", "uint", input_size, "readonly"
        )
        self.emit(f"layout(std430, binding = {input_binding}) "
                  f"readonly buffer InputData {{")
        self.emit(f"    uint input_regs[];")
        self.emit("};")
        self.emit("")

        # Output SSBO
        output_binding = self.alloc_binding(
            "output_data", "uint", input_size, "writeonly"
        )
        self.emit(f"layout(std430, binding = {output_binding}) "
                  f"writeonly buffer OutputData {{")
        self.emit(f"    uint output_regs[];")
        self.emit("};")
        self.emit("")

        # Modular arithmetic helpers (inlined for branch-free execution)
        self._emit_mod_helpers()

        # Main function
        self.emit("void main() {")
        self._indent += 1

        self.emit("uint gid = gl_GlobalInvocationID.x;")
        self.emit("")

        # Walk IR in topological order and emit operations
        for name, node in prog.topo_order():
            self._emit_node(name, node)

        self._indent -= 1
        self.emit("}")

        source = "\n".join(self._lines)

        return GLSLShaderBundle(
            glsl_source=source,
            root_table=self.roots,
            buffers=self._buffers,
            workgroup_size=(self.workgroup_x, 1, 1),
            field_p=self.p,
            field_q=self.q,
        )

    # ── Helper functions emitted into the shader ─────────────────────

    def _emit_mod_helpers(self) -> None:
        self.emit(f"// Modular arithmetic (branch-free, uint only)")
        self.emit(f"// All intermediate results fit in uint64 for p < 2^16")
        self.emit(f"const uint P = {self.p}u;")
        self.emit(f"const uint Q = {self.q}u;")
        self.emit(f"const uint M = {self.m}u;  // order of F_p*")
        self.emit("")
        # mul_mod_p: (a * b) % p using 64-bit intermediate
        # GLSL 4.50 doesn't have native uint64, so for p < 2^16 we
        # stay in uint32. For larger p, would need packUint2x32.
        self.emit("uint mul_mod_p(uint a, uint b) {")
        self.emit("    return (a * b) % P;")
        self.emit("}")
        self.emit("")
        self.emit("uint mul_mod_q(uint a, uint b) {")
        self.emit("    return (a * b) % Q;")
        self.emit("}")
        self.emit("")
        self.emit("uint add_mod_p(uint a, uint b) {")
        self.emit("    return (a + b) % P;")
        self.emit("}")
        self.emit("")

    # ── Per-node code generation ─────────────────────────────────────

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
        elif isinstance(node, IRCatalyticRegion):
            self.emit(f"// catalytic region '{name}' "
                      f"(restore: {node.catalytic_regs})")
        elif isinstance(node, IRCycloPhiPoly):
            self._emit_cyclo_phi(name, node)
        elif isinstance(node, IRMatPow):
            self._emit_matpow_stub(name, node)

    def _emit_embed(self, name: str, node: IREmbed) -> None:
        var = self.fresh_var("emb")
        self._var_map[name] = var
        if node.psi == 0:
            self.emit(f"uint {var} = 1u;  // embed({node.index}, 0) = omega^0")
        else:
            j = node.psi % self.m
            self.emit(f"uint {var} = omega[{j}];  "
                      f"// embed({node.index}, {node.psi})")

    def _emit_primitive_roots(self, name: str, node: IRPrimitiveRoots) -> None:
        # Root table is already in the SSBO; just note it
        self._var_map[name] = "/* root_table */"
        self.emit(f"// roots({node.ir_field.p if node.ir_field else '?'}) "
                  f"loaded in SSBO binding 0")

    def _emit_rotate(self, name: str, node: IRRotate) -> None:
        var = self.fresh_var("rot")
        j = node.j % self.m
        src_var = self._resolve_parent(node, 0)
        self.emit(f"uint {var} = mul_mod_p({src_var}, omega[{j}]);  "
                  f"// rotate by omega^{j}")
        self._var_map[name] = var

    def _emit_poly(self, name: str, node: IRPoly) -> None:
        """Emit fully unrolled Horner evaluation.

        For coefficients [c0, c1, c2, ..., cd] (ascending order):
          result = c_d
          result = result * x + c_{d-1}  (mod q)
          ...
          result = result * x + c_0       (mod q)
          result = result % p             (reduce to output field)
        """
        var = self.fresh_var("poly")
        x_var = self._resolve_parent(node, 0)

        # Horner: process coefficients in descending order
        coeffs_desc = list(reversed(node.coeffs))

        self.emit(f"// polynomial degree {node.degree}, "
                  f"Horner in F_{self.q}")
        self.emit(f"uint {var} = {coeffs_desc[0]}u;")

        for i, c in enumerate(coeffs_desc[1:], 1):
            self.emit(f"{var} = (mul_mod_q({var}, {x_var}) + {c}u) % Q;  "
                      f"// Horner step {i}/{len(coeffs_desc)-1}")

        # Reduce to output field
        self.emit(f"{var} = {var} % P;  // reduce to F_{self.p}")
        self._var_map[name] = var

    def _emit_measure(self, name: str, node: IRMeasure) -> None:
        var = self.fresh_var("meas")
        src_var = self._resolve_parent(node, 0)

        if node.mode == 'mod_p':
            self.emit(f"uint {var} = {src_var} % P;  // measure mod_p")
        elif node.mode == 'real':
            self.emit(f"uint {var} = {src_var};  // measure real")
        elif node.mode == 'mag':
            self.emit(f"uint {var} = {src_var};  // measure mag (F_p: identity)")
        else:
            self.emit(f"uint {var} = {src_var};  // measure {node.mode}")

        self.emit(f"output_regs[gid] = {var};")
        self._var_map[name] = var

    def _emit_compose(self, name: str, node: IRCompose) -> None:
        # Composition in the shader is implicit — the topological ordering
        # ensures parents are computed before children. Just alias.
        if node.parents:
            last_parent = node.parents[-1]
            parent_var = self._var_map.get(last_parent.name, "0u")
            self._var_map[name] = parent_var
            self.emit(f"// compose '{name}' = {parent_var}")

    def _emit_cyclo_phi(self, name: str, node: IRCycloPhiPoly) -> None:
        """Emit cyclotomic polynomial Phi_n(x) as constant array."""
        var = self.fresh_var("phi")
        coeffs = node.poly_coeffs
        # Emit as constant array for FIR-style convolution use
        self.emit(f"// Phi_{node.n}(x): {len(coeffs)} coefficients")
        arr_name = f"phi_{node.n}_coeffs"
        coeffs_str = ", ".join(f"{c}u" for c in coeffs)
        self.emit(f"const uint {arr_name}[{len(coeffs)}] = "
                  f"uint[]({coeffs_str});")
        self._var_map[name] = arr_name

    def _emit_matpow_stub(self, name: str, node: IRMatPow) -> None:
        """Emit a stub for matrix powering.

        Full matpow in GLSL requires tiled matrix multiply dispatch
        which depends on the matrix dimensions. This emits the setup
        and documents the required dispatch structure.
        """
        self.emit(f"// === MatPow stub: {node.matrix_name}^{node.d} ===")
        self.emit(f"// delta = {node.delta}, "
                  f"epsilon = {node.epsilon}")
        self.emit(f"// Full implementation requires:")
        self.emit(f"//   - Tiled matmul kernel (separate dispatch)")
        self.emit(f"//   - Base-delta digit loop unrolled to "
                  f"{len(__import__('cmtz.field', fromlist=['to_base']).to_base(node.d, node.delta))} stages")
        self.emit(f"//   - Each stage: polynomial eval (this shader) "
                  f"+ matmul (dispatch)")
        var = self.fresh_var("matpow")
        self.emit(f"uint {var} = 0u;  // placeholder — see dispatch code")
        self._var_map[name] = var

    # ── Helpers ──────────────────────────────────────────────────────

    def _resolve_parent(self, node: IRNode, idx: int) -> str:
        """Get the GLSL variable name for the idx-th parent."""
        if idx < len(node.parents):
            parent_name = node.parents[idx].name
            return self._var_map.get(parent_name, "0u")
        return "input_regs[gid]"


# ── Public API ───────────────────────────────────────────────────────────────

def emit_glsl(prog: IRProgram, workgroup_x: int = 64) -> GLSLShaderBundle:
    """Emit a GLSL compute shader from an IR program.

    Args:
        prog: Lowered and analyzed IR program
        workgroup_x: Workgroup X dimension (default 64)

    Returns:
        GLSLShaderBundle with source, root table, and buffer metadata

    Raises:
        ValueError if program has no field annotation
    """
    if prog.ir_field is None:
        raise ValueError(
            "Program has no field annotation — cannot emit GLSL"
        )

    emitter = GLSLEmitter(prog.ir_field, workgroup_x=workgroup_x)
    return emitter.generate(prog)

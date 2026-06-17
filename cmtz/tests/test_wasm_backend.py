"""Tests for the WebAssembly Text Format (WAT) backend."""

import pytest
from cmtz.compiler import compile_program
from cmtz.field import IRField
from cmtz.ir_nodes import IRProgram, IREmbed, IRPrimitiveRoots
from cmtz.backends.wasm_backend import emit_wasm, WasmBundle


_SIMPLE = """
field(7);
roots(7) as omega;
embed(0, 0);
rotate(embed_0, embed_0, 3);
measure(rot_embed_0_embed_0, mod_p) as result;
"""

_REAL_ADD = """
field(17, 19);
roots(17) as omega17;
embed(0, 0);
embed(1, 4);
add(embed_0, embed_1) as superposed;
measure(superposed, mod_p) as sum_val;
"""

_PHASOR = """
cfield(7);
cembed(0, 0);
cembed(1, 2);
conj(cembed_2) as conj_2;
magsq(cembed_2) as magsq_2;
measure(magsq_2, mod_p) as norm;
"""


class TestWasmBundle:
    def test_compile_wasm_backend(self):
        result = compile_program(_SIMPLE, backend='wasm')
        assert result.ok, f"Errors: {result.errors}"
        out = result.backend_output
        assert 'wat_source' in out
        assert 'root_table' in out
        assert 'layout' in out
        assert 'assemble_command' in out

    def test_wat_is_valid_module(self):
        result = compile_program(_SIMPLE, backend='wasm')
        wat = result.backend_output['wat_source']
        assert wat.startswith('(module')
        assert wat.rstrip().endswith(')')
        # Balanced parentheses
        depth = sum(1 if c == '(' else -1 if c == ')' else 0 for c in wat)
        assert depth == 0, "WAT has unbalanced parentheses"

    def test_wat_has_memory_and_data(self):
        result = compile_program(_SIMPLE, backend='wasm')
        wat = result.backend_output['wat_source']
        assert '(memory 1)' in wat
        assert '(data (i32.const 0)' in wat

    def test_wat_exports_memory_and_compute(self):
        result = compile_program(_SIMPLE, backend='wasm')
        wat = result.backend_output['wat_source']
        assert '(export "memory" (memory 0))' in wat
        assert '(export "compute"' in wat

    def test_wat_has_mul_helper(self):
        result = compile_program(_SIMPLE, backend='wasm')
        wat = result.backend_output['wat_source']
        assert '(func $mul_p' in wat
        assert 'i64.mul' in wat          # widening multiply
        assert 'i64.extend_i32_u' in wat

    def test_wat_has_add_helper(self):
        result = compile_program(_SIMPLE, backend='wasm')
        wat = result.backend_output['wat_source']
        assert '(func $add_p' in wat

    def test_root_table_values_F7(self):
        """Root table for F_7 (g=3, m=6): [1, 3, 2, 6, 4, 5]."""
        result = compile_program(_SIMPLE, backend='wasm')
        roots = result.backend_output['root_table']
        from cmtz.field import primitive_roots_Fp
        expected = list(primitive_roots_Fp(7))
        assert roots == expected

    def test_layout_offsets(self):
        result = compile_program(_SIMPLE, backend='wasm')
        layout = result.backend_output['layout']
        assert layout['root_table_offset'] == 0
        assert layout['root_table_bytes'] == 6 * 4   # m=6 for F_7
        assert layout['output_offset'] == 6 * 4
        assert layout['measure_count'] == 1

    def test_assemble_command(self):
        result = compile_program(_SIMPLE, backend='wasm')
        cmd = result.backend_output['assemble_command']
        assert 'wat2wasm' in cmd
        assert '.wasm' in cmd

    def test_embed_emits_load_or_const(self):
        """embed(0, 0) emits i32.const 1; embed(_, j>0) emits i32.load."""
        src = "field(7); embed(0, 0); embed(1, 3); measure(embed_0, mod_p) as r;"
        result = compile_program(src, backend='wasm')
        wat = result.backend_output['wat_source']
        # psi=0 special-cased to i32.const 1
        assert 'i32.const 1' in wat
        # psi=3 → byte offset 12 = 3*4
        assert 'i32.load (i32.const 12)' in wat

    def test_rotate_emits_mul_p_call(self):
        result = compile_program(_SIMPLE, backend='wasm')
        wat = result.backend_output['wat_source']
        assert 'call $mul_p' in wat

    def test_measure_writes_to_output_area(self):
        result = compile_program(_SIMPLE, backend='wasm')
        wat = result.backend_output['wat_source']
        # F_7, m=6, output_offset=24; first measure at byte 24
        assert 'i32.store (i32.const 24)' in wat

    def test_add_real_emits_add_p(self):
        result = compile_program(_REAL_ADD, backend='wasm')
        assert result.ok, result.errors
        wat = result.backend_output['wat_source']
        assert 'call $add_p' in wat

    def test_phasor_has_magsq_helper(self):
        result = compile_program(_PHASOR, backend='wasm')
        assert result.ok, result.errors
        wat = result.backend_output['wat_source']
        assert '(func $magsq' in wat

    def test_phasor_complex_locals(self):
        """Complex registers produce _re / _im local pairs."""
        result = compile_program(_PHASOR, backend='wasm')
        wat = result.backend_output['wat_source']
        assert '$cembed_1_re' in wat
        assert '$cembed_1_im' in wat
        assert '$cembed_2_re' in wat
        assert '$cembed_2_im' in wat
        assert '$conj_2_re' in wat
        assert '$conj_2_im' in wat

    def test_phasor_conjugate_negates_im(self):
        result = compile_program(_PHASOR, backend='wasm')
        wat = result.backend_output['wat_source']
        # Conjugate uses (p - im) % p
        assert 'i32.sub (i32.const 7)' in wat

    def test_phasor_wat_balanced_parens(self):
        result = compile_program(_PHASOR, backend='wasm')
        wat = result.backend_output['wat_source']
        depth = sum(1 if c == '(' else -1 if c == ')' else 0 for c in wat)
        assert depth == 0

    def test_no_field_raises(self):
        prog = IRProgram(ir_field=None)
        with pytest.raises(ValueError, match="no field annotation"):
            emit_wasm(prog)

    def test_emit_wasm_direct(self):
        """emit_wasm() works directly on an IRProgram."""
        field = IRField(p=7, q=7)
        prog = IRProgram(ir_field=field)
        roots_node = IRPrimitiveRoots.build(ir_field=field, name='omega')
        prog.add_node(roots_node)
        embed = IREmbed(name='e0', ir_field=field, index=0, psi=0)
        prog.add_node(embed)
        bundle = emit_wasm(prog)
        assert isinstance(bundle, WasmBundle)
        assert '(module' in bundle.wat_source
        assert bundle.field_p == 7

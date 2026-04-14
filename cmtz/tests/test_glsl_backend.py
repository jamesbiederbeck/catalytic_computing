"""Tests for GLSL compute shader backend."""

import pytest
from cmtz.compiler import compile_program
from cmtz.field import IRField
from cmtz.ir_nodes import IRProgram, IREmbed, IRRotate, IRPrimitiveRoots
from cmtz.backends.glsl_backend import emit_glsl, GLSLShaderBundle


class TestGLSLEmission:
    def test_simple_program(self):
        """Emit GLSL for a minimal roots + rotate + measure program."""
        src = """
        field(7);
        roots(7) as omega;
        embed(0, 0);
        rotate(embed_0, embed_0, 3);
        measure(rot_embed_0_embed_0, mod_p) as result;
        """
        result = compile_program(src, backend='glsl')
        assert result.ok, f"Errors: {result.errors}"

        out = result.backend_output
        assert 'glsl_source' in out
        assert 'root_table' in out
        assert 'buffers' in out

    def test_shader_has_version(self):
        src = "field(7); embed(0, 0); measure(embed_0, mod_p) as result;"
        result = compile_program(src, backend='glsl')
        assert '#version 450' in result.backend_output['glsl_source']

    def test_shader_has_mod_constants(self):
        src = "field(7); embed(0, 0); measure(embed_0, mod_p) as result;"
        result = compile_program(src, backend='glsl')
        glsl = result.backend_output['glsl_source']
        assert 'const uint P = 7u;' in glsl
        assert 'const uint Q = 7u;' in glsl
        assert 'const uint M = 6u;' in glsl

    def test_shader_has_root_table_ssbo(self):
        src = "field(7); roots(7) as omega; embed(0, 0); measure(embed_0, mod_p) as r;"
        result = compile_program(src, backend='glsl')
        glsl = result.backend_output['glsl_source']
        assert 'buffer RootTable' in glsl
        assert 'omega[6]' in glsl  # m = p-1 = 6

    def test_root_table_values(self):
        src = "field(7); embed(0, 0); measure(embed_0, mod_p) as r;"
        result = compile_program(src, backend='glsl')
        roots = result.backend_output['root_table']
        assert len(roots) == 6
        assert set(roots) == {1, 2, 3, 4, 5, 6}
        assert roots[0] == 1

    def test_rotate_emits_mul_mod(self):
        src = """
        field(7);
        roots(7) as omega;
        embed(0, 0);
        rotate(embed_0, embed_0, 3);
        measure(rot_embed_0_embed_0, mod_p) as r;
        """
        result = compile_program(src, backend='glsl')
        glsl = result.backend_output['glsl_source']
        assert 'mul_mod_p' in glsl
        assert 'omega[3]' in glsl

    def test_poly_emits_horner(self):
        """Verify polynomial nodes emit unrolled Horner steps."""
        from cmtz.ir_nodes import IRPoly
        from cmtz.backends.glsl_backend import GLSLEmitter

        field = IRField(p=7, q=7)
        emitter = GLSLEmitter(field)
        prog = IRProgram(ir_field=field)

        embed = IREmbed(name="x", ir_field=field, index=0, psi=0)
        prog.add_node(embed)

        poly = IRPoly(
            name="p1", ir_field=field,
            coeffs=[1, 2, 3],  # 1 + 2x + 3x^2
            degree=2,
            parents=[embed],
        )
        prog.add_node(poly)

        bundle = emitter.generate(prog)
        assert 'Horner step' in bundle.glsl_source
        assert 'mul_mod_q' in bundle.glsl_source

    def test_buffer_bindings(self):
        src = "field(7); embed(0, 0); measure(embed_0, mod_p) as r;"
        result = compile_program(src, backend='glsl')
        buffers = result.backend_output['buffers']

        # Should have at least: root table, input, output
        assert len(buffers) >= 3
        names = [b['name'] for b in buffers]
        assert 'roots' in names
        assert 'input_data' in names
        assert 'output_data' in names

    def test_workgroup_size(self):
        src = "field(7); embed(0, 0); measure(embed_0, mod_p) as r;"
        result = compile_program(src, backend='glsl')
        ws = result.backend_output['workgroup_size']
        assert ws == [64, 1, 1]

    def test_spirv_command(self):
        src = "field(7); embed(0, 0); measure(embed_0, mod_p) as r;"
        result = compile_program(src, backend='glsl')
        cmd = result.backend_output['spirv_command']
        assert 'glslangValidator' in cmd
        assert '-V' in cmd

    def test_no_branches_in_output(self):
        """The shader must be fully branch-free per the IR invariant."""
        src = """
        field(7);
        roots(7) as omega;
        embed(0, 0);
        rotate(embed_0, embed_0, 2);
        measure(rot_embed_0_embed_0, mod_p) as r;
        """
        result = compile_program(src, backend='glsl')
        glsl = result.backend_output['glsl_source']
        # No if/else/while/for in the generated shader body
        # (only in comments is acceptable)
        lines = [l for l in glsl.split('\n')
                 if not l.strip().startswith('//')]
        body = '\n'.join(lines)
        assert ' if(' not in body and ' if (' not in body
        assert 'while' not in body
        assert 'for(' not in body and 'for (' not in body

    def test_larger_field(self):
        src = "field(17, 293); embed(0, 0); measure(embed_0, mod_p) as r;"
        result = compile_program(src, backend='glsl')
        assert result.ok
        glsl = result.backend_output['glsl_source']
        assert 'const uint P = 17u;' in glsl
        assert 'const uint Q = 293u;' in glsl


class TestDirectEmit:
    """Test the emit_glsl function directly."""

    def test_empty_program_with_field(self):
        prog = IRProgram(ir_field=IRField(p=5, q=5))
        bundle = emit_glsl(prog)
        assert isinstance(bundle, GLSLShaderBundle)
        assert bundle.field_p == 5
        assert len(bundle.root_table) == 4  # m = p-1 = 4

    def test_no_field_raises(self):
        prog = IRProgram()
        with pytest.raises(ValueError, match="no field"):
            emit_glsl(prog)

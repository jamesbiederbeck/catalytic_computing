"""Tests for parser, catalytic verify, cost propagation, and end-to-end."""

import pytest
from cmtz.lexer import tokenize
from cmtz.parser import Parser, ParseError
from cmtz.ast_nodes import (
    Program, FieldDecl, EmbedStmt, RotateStmt, RootsStmt,
    MatPowStmt, CatalyticStmt, MeasureStmt,
)
from cmtz.elaborator import Elaborator, ElaborationError
from cmtz.field import IRField
from cmtz.ir_nodes import IRCompose, IRRegion, IRCatalyticRegion, IRPrimitiveRoots
from cmtz.lowering import lower
from cmtz.analysis.catalytic_verify import verify_catalytic, CatalyticViolationError
from cmtz.analysis.cost_propagation import propagate_costs
from cmtz.compiler import compile_program


# ── Parser tests ─────────────────────────────────────────────────────────────

class TestParser:
    def _parse(self, src: str) -> Program:
        tokens = tokenize(src)
        return Parser(tokens).parse()

    def test_field_decl_two_args(self):
        prog = self._parse("field(17, 289);")
        assert prog.field_decl is not None
        assert prog.field_decl.p == 17
        assert prog.field_decl.q == 289

    def test_field_decl_one_arg(self):
        prog = self._parse("field(7);")
        assert prog.field_decl.p == 7
        assert prog.field_decl.q is None

    def test_embed(self):
        prog = self._parse("embed(0, 3);")
        stmt = prog.statements[0]
        assert isinstance(stmt, EmbedStmt)
        assert stmt.index == 0
        assert stmt.psi == 3

    def test_rotate_int_j(self):
        prog = self._parse("embed(0, 0); rotate(embed_0, embed_1, 5);")
        stmt = prog.statements[1]
        assert isinstance(stmt, RotateStmt)
        assert stmt.j == 5  # integer exponent, not float

    def test_roots(self):
        prog = self._parse("roots(17) as omega17;")
        stmt = prog.statements[0]
        assert isinstance(stmt, RootsStmt)
        assert stmt.p == 17
        assert stmt.name == "omega17"

    def test_matpow(self):
        prog = self._parse("field(17); matpow(M, 8, 0.5) as Mpow8;")
        stmt = prog.statements[0]
        assert isinstance(stmt, MatPowStmt)
        assert stmt.d == 8
        assert stmt.epsilon == 0.5

    def test_catalytic(self):
        src = """
        field(17);
        embed(0, 0);
        catalytic {
            embed(1, 3);
        } restoring (r0, r1);
        """
        prog = self._parse(src)
        cat = prog.statements[1]
        assert isinstance(cat, CatalyticStmt)
        assert len(cat.body) == 1
        assert cat.restoring == ['r0', 'r1']

    def test_measure_mod_p(self):
        prog = self._parse("measure(x, mod_p) as result;")
        stmt = prog.statements[0]
        assert isinstance(stmt, MeasureStmt)
        assert stmt.mode == 'mod_p'

    def test_full_v2_program(self):
        src = """
        field(17, 289);
        roots(17) as omega17;
        embed(0, 0);
        embed(1, 4);
        catalytic {
            matpow(M, 8, 0.5) as Mpow8;
        } restoring (r0, r1, r2, r3);
        measure(Mpow8, mod_p) as result;
        """
        prog = self._parse(src)
        assert prog.field_decl.p == 17
        assert len(prog.statements) == 5

    def test_parse_error_bad_token(self):
        with pytest.raises(ParseError):
            self._parse("badkeyword(1);")


# ── Elaboration tests ────────────────────────────────────────────────────────

class TestElaborator:
    def test_field_resolution(self):
        tokens = tokenize("field(7);")
        ast = Parser(tokens).parse()
        elab = Elaborator()
        elab.elaborate(ast)
        assert elab.ir_field == IRField(p=7, q=7)

    def test_field_with_q(self):
        tokens = tokenize("field(7, 97);")
        ast = Parser(tokens).parse()
        elab = Elaborator()
        elab.elaborate(ast)
        assert elab.ir_field == IRField(p=7, q=97)

    def test_non_prime_p_raises(self):
        tokens = tokenize("field(6);")
        ast = Parser(tokens).parse()
        elab = Elaborator()
        with pytest.raises(ElaborationError, match="not prime"):
            elab.elaborate(ast)

    def test_roots_infers_field(self):
        tokens = tokenize("roots(7) as r;")
        ast = Parser(tokens).parse()
        elab = Elaborator()
        elab.elaborate(ast)
        assert elab.ir_field is not None
        assert elab.ir_field.p == 7


# ── Catalytic verification tests ─────────────────────────────────────────────

class TestCatalyticVerify:
    def test_clean_program_passes(self):
        """A catalytic region with zero net delta should pass."""
        src = """
        field(7);
        embed(0, 0);
        catalytic {
            embed(1, 3);
        } restoring (r0);
        """
        result = compile_program(src, backend='python')
        assert result.analysis_report.get('catalytic_verified', False)

    def test_region_exists_in_ir(self):
        """catalytic {} lowers to IRRegion (advisory), not IRCatalyticRegion."""
        src = """
        field(7);
        catalytic {
            embed(0, 0);
        } restoring (r0, r1);
        """
        result = compile_program(src, backend='python')
        region_nodes = [
            n for _, n in result.ir_program.nodes()
            if isinstance(n, IRRegion)
        ]
        strict_nodes = [
            n for _, n in result.ir_program.nodes()
            if isinstance(n, IRCatalyticRegion)
        ]
        assert len(region_nodes) == 1
        assert region_nodes[0].catalytic_regs == ['r0', 'r1']
        assert len(strict_nodes) == 0

    def test_interrupt_exists_in_ir(self):
        """interrupt {} lowers to IRCatalyticRegion (strict)."""
        src = """
        field(7);
        embed(0, 0);
        interrupt {
            embed(1, 2);
        } restoring (r0);
        """
        result = compile_program(src, backend='python')
        strict_nodes = [
            n for _, n in result.ir_program.nodes()
            if isinstance(n, IRCatalyticRegion)
        ]
        region_nodes = [
            n for _, n in result.ir_program.nodes()
            if type(n) is IRRegion
        ]
        assert len(strict_nodes) == 1
        assert strict_nodes[0].catalytic_regs == ['r0']
        assert len(region_nodes) == 0

    def test_interrupt_clobber_raises(self):
        """IREmbed inside interrupt {} targeting a listed register is a violation."""
        src = """
        field(7);
        interrupt {
            embed(0, 3);
        } restoring (embed_0);
        """
        from cmtz.analysis.catalytic_verify import CatalyticViolationError
        result = compile_program(src, backend='python')
        assert not result.ok
        assert any('Catalytic verification' in e for e in result.errors)

    def test_catalytic_advisory_allows_embed(self):
        """IREmbed inside catalytic {} targeting a listed register is fine — advisory."""
        src = """
        field(7);
        catalytic {
            embed(0, 3);
        } restoring (embed_0);
        """
        result = compile_program(src, backend='python')
        assert result.ok

    def test_interrupt_clean_passes(self):
        """interrupt {} with no clobber verifies cleanly."""
        src = """
        field(7);
        embed(0, 0);
        interrupt {
            embed(1, 2);
        } restoring (embed_0);
        """
        result = compile_program(src, backend='python')
        assert result.ok
        assert result.analysis_report['catalytic_verified']
        assert result.analysis_report['catalytic_deltas'] == [
            {'register': 'embed_0', 'delta': '0'}
        ]

    def test_advisory_count_reported(self):
        """Compiler reports advisory region count separately from verified."""
        src = """
        field(7);
        catalytic {
            embed(0, 0);
        } restoring (r0);
        catalytic {
            embed(1, 1);
        } restoring (r1);
        interrupt {
            embed(2, 2);
        } restoring (r2);
        """
        result = compile_program(src, backend='python')
        assert result.analysis_report['catalytic_advisory_count'] == 2
        assert result.analysis_report['catalytic_verified']


# ── Cost propagation tests ───────────────────────────────────────────────────

class TestCostPropagation:
    def test_compose_costs(self):
        """Verify Lemma 2.1 cost accounting on composition."""
        src = """
        field(7);
        embed(0, 0);
        embed(1, 0);
        compose(embed_0, embed_1);
        """
        result = compile_program(src, backend='python')
        # Find the compose node
        compose_nodes = [
            n for _, n in result.ir_program.nodes()
            if isinstance(n, IRCompose)
        ]
        assert len(compose_nodes) >= 1
        comp = compose_nodes[0]
        # Both parents have recursive_calls=0, so max(1, 0)*max(1, 0) = 1
        assert comp.recursive_calls == 1


# ── End-to-end compilation tests ─────────────────────────────────────────────

class TestEndToEnd:
    def test_simple_embed_measure(self):
        src = """
        field(7);
        embed(0, 0);
        measure(embed_0, mod_p) as result;
        """
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"
        assert 'result' in result.backend_output

    def test_roots_and_rotate(self):
        src = """
        field(7);
        roots(7) as omega;
        embed(0, 0);
        rotate(embed_0, embed_0, 3);
        measure(rot_embed_0_embed_0, mod_p) as result;
        """
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"

    def test_analog_backend(self):
        src = """
        field(7);
        roots(7) as omega;
        embed(0, 0);
        measure(embed_0, mod_p) as result;
        """
        result = compile_program(src, backend='analog')
        assert result.ok, f"Errors: {result.errors}"
        assert result.backend_output['field_p'] == 7
        assert result.backend_output['root_order'] == 6
        assert len(result.backend_output['phase_offsets']) == 6

    def test_optimization_stats(self):
        src = """
        field(7);
        embed(0, 0);
        measure(embed_0, mod_p) as result;
        """
        result = compile_program(src, backend='python', optimize=True)
        assert 'rotations_fused' in result.optimization_stats


# ── Matrix powering tests ────────────────────────────────────────────────────

class TestMatPow:
    def test_identity_matrix(self):
        """I^d = I for any d."""
        from cmtz.backends.python_ref import PythonBackend
        from cmtz.ir_nodes import IRMatPow

        field = IRField(p=7, q=7)
        be = PythonBackend(field)

        I = [[1, 0], [0, 1]]
        node = IRMatPow(
            name="test", ir_field=field,
            matrix_name="I", d=10, epsilon=0.5, n_dim=2,
        )
        result = be.eval_matpow_matrix(node, I)
        assert result == [[1, 0], [0, 1]]

    def test_small_matrix_power(self):
        """Verify M^3 mod 7 for a small matrix."""
        from cmtz.backends.python_ref import PythonBackend
        from cmtz.ir_nodes import IRMatPow

        field = IRField(p=7, q=7)
        be = PythonBackend(field)

        M = [[2, 1], [1, 3]]
        node = IRMatPow(
            name="test", ir_field=field,
            matrix_name="M", d=3, epsilon=0.5, n_dim=2,
        )
        result = be.eval_matpow_matrix(node, M)

        # Verify by naive multiplication
        def matmul_mod(A, B, p):
            n = len(A)
            C = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % p
            return C

        M2 = matmul_mod(M, M, 7)
        M3 = matmul_mod(M2, M, 7)
        assert result == M3

    def test_base_delta_decomposition(self):
        """Verify to_base decomposition reconstructs the original exponent."""
        from cmtz.field import to_base

        d = 100
        delta = 8  # 2^(3/1) for ε=1
        digits = to_base(d, delta)
        reconstructed = sum(a * delta**i for i, a in enumerate(digits))
        assert reconstructed == d


# ── Phasor network (v4) tests ─────────────────────────────────────────────────

class TestPhasorNetwork:
    def test_cfield_declaration(self):
        """cfield(7) auto-computes c and q."""
        src = "cfield(7);"
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"
        field = result.ir_program.ir_field
        assert field.is_complex
        assert field.p == 7

    def test_cfield_q_auto_computed(self):
        """Auto-computed q for cfield(7) should be a prime >= 49."""
        src = "cfield(7);"
        result = compile_program(src, backend='python')
        assert result.ok
        field = result.ir_program.ir_field
        import sympy
        assert sympy.isprime(field.q)
        assert field.q >= 7 * 7

    def test_cembed_lowers_to_ir(self):
        """cembed(0, 1) produces an IRComplexEmbed node."""
        from cmtz.ir_nodes import IRComplexEmbed
        src = "cfield(7); cembed(0, 1);"
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"
        nodes = [n for _, n in result.ir_program.nodes()
                 if isinstance(n, IRComplexEmbed)]
        assert len(nodes) == 1
        assert nodes[0].re_psi == 0
        assert nodes[0].im_psi == 1

    def test_cembed_without_cfield_raises(self):
        """cembed without cfield should fail during elaboration."""
        src = "field(7); cembed(0, 1);"
        result = compile_program(src, backend='python')
        assert not result.ok
        assert any("cfield" in e for e in result.errors)

    def test_add_two_real_registers(self):
        """add(a, b) on real field registers works end-to-end."""
        src = """
        field(7);
        embed(0, 0);
        embed(1, 1);
        add(embed_0, embed_1) as ab;
        measure(ab, mod_p) as result;
        """
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"
        assert 'result' in result.backend_output

    def test_add_lowers_to_ir(self):
        """add(a, b) produces an IRAdd node."""
        from cmtz.ir_nodes import IRAdd
        src = """
        field(7);
        embed(0, 0);
        embed(1, 0);
        add(embed_0, embed_1) as ab;
        """
        result = compile_program(src, backend='python')
        assert result.ok
        add_nodes = [n for _, n in result.ir_program.nodes() if isinstance(n, IRAdd)]
        assert len(add_nodes) == 1

    def test_conj_lowers_to_ir(self):
        """conj(a) produces an IRConjugate node."""
        from cmtz.ir_nodes import IRConjugate
        src = """
        cfield(7);
        cembed(0, 1);
        conj(cembed_1) as c;
        """
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"
        conj_nodes = [n for _, n in result.ir_program.nodes() if isinstance(n, IRConjugate)]
        assert len(conj_nodes) == 1

    def test_magsq_lowers_to_ir(self):
        """magsq(a) produces an IRMagnitudeSq node."""
        from cmtz.ir_nodes import IRMagnitudeSq
        src = """
        cfield(7);
        cembed(0, 1);
        magsq(cembed_1) as ms;
        """
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"
        msq_nodes = [n for _, n in result.ir_program.nodes() if isinstance(n, IRMagnitudeSq)]
        assert len(msq_nodes) == 1

    def test_magsq_type_lowers_to_real(self):
        """IRMagnitudeSq output field should be real (c=None)."""
        from cmtz.ir_nodes import IRMagnitudeSq
        src = """
        cfield(7);
        cembed(0, 1);
        magsq(cembed_1) as ms;
        """
        result = compile_program(src, backend='python')
        assert result.ok
        msq = next(n for _, n in result.ir_program.nodes() if isinstance(n, IRMagnitudeSq))
        assert not msq.ir_field.is_complex

    def test_full_phasor_pipeline(self):
        """cembed → add → magsq → measure end-to-end."""
        src = """
        cfield(7);
        cembed(0, 1);
        cembed(2, 0);
        add(cembed_1, cembed_2) as C;
        magsq(C) as power;
        measure(power, mod_p) as result;
        """
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"
        assert 'result' in result.backend_output
        # result should be a value in F_7
        val = result.backend_output['result']
        assert 0 <= val < 7

    def test_phasor_result_correct(self):
        """Verify numerical correctness for a known phasor computation.

        cfield(7): c = find_nonresidue(7), roots = primitive_roots_Fp(7)
        cembed(0, 1): A = (roots[0], roots[1]) = (1, g) where g=generator
        magsq(A) = 1² - g²·c mod 7
        """
        from cmtz.field import find_nonresidue, primitive_roots_Fp, magsq_Fp2
        src = """
        cfield(7);
        cembed(0, 1);
        magsq(cembed_1) as ms;
        measure(ms, mod_p) as result;
        """
        result = compile_program(src, backend='python')
        assert result.ok, f"Errors: {result.errors}"

        # Compute expected value manually
        c = find_nonresidue(7)
        roots = primitive_roots_Fp(7)
        A = (roots[0], roots[1])
        expected = magsq_Fp2(A, 7, c)
        assert result.backend_output['result'] == expected

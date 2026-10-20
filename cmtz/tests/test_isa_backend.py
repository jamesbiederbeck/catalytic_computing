"""cmtz-core backend and allocated-code catalytic verifier."""

import random
from pathlib import Path

import pytest

from cmtz import compile_program
from cmtz.analysis.isa_catalytic import IsaCatalyticError, verify_handler
from cmtz.backends.isa_backend import emit_isa
from cmtz.backends.python_ref import PythonBackend
from cmtz.backends.verilog_backend import evaluate_all
from cmtz.ir_nodes import IRMatPow
from cmtz.isa import CoreConfig, ProgramImage, assemble, run_image

CORE = CoreConfig(17, 19)
from cmtz.isa import sim as S
from cmtz.isa.encoding import decode, written_registers

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"
REAL = [f for f in sorted(EXAMPLES.glob("*.cmtz"))
        if "cfield" not in f.read_text()]


def ir(path_or_src):
    src = path_or_src.read_text() if isinstance(path_or_src, Path) else path_or_src
    r = compile_program(src, backend="python")
    assert r.ok, r.errors
    return r.ir_program


def matpows(prog):
    return [n for _, n in prog.nodes() if isinstance(n, IRMatPow)]


def waiti_count(bundle) -> int:
    """Retired count at which main stalls on WAITI."""
    m = run_image(bundle.core, bundle.image,
                  {n: 0 for n in bundle.inputs},
                  {k: [[0] * n for _ in range(n)]
                   for k, n in bundle.matrices_in.items()})
    assert m.waiting
    return m.retired


@pytest.mark.parametrize("path", REAL, ids=lambda p: p.stem)
def test_example_matches_reference(path):
    prog = ir(path)
    p = prog.ir_field.p
    dims = {n.matrix_name: 2 for n in matpows(prog)}
    b = emit_isa(prog, matrix_dims=dims)
    rng = random.Random(path.stem)
    stall = waiti_count(b) if b.image.handler is not None else None
    for _ in range(40):
        ins = {n: rng.randrange(p) for n in b.inputs}
        mats = {k: [[rng.randrange(p) for _ in range(n)] for _ in range(n)]
                for k, n in b.matrices_in.items()}
        irq = (rng.randrange(stall + 1),) if stall is not None else ()
        m = run_image(b.core, b.image, ins, mats, irq_at=irq,
                      scramble=rng.randrange(1 << 30))
        assert m.status == S.HALTED, (m.status, m.error)
        exp = evaluate_all(prog, prog.ir_field, ins)
        assert m.outputs() == {o: exp[o] % p for o in b.outputs}
        for node in matpows(prog):
            want = PythonBackend(prog.ir_field).eval_matpow_matrix(
                node, mats[node.matrix_name])
            assert m.read_matrix(node.name) == want


def test_complex_rejected():
    path = next(f for f in EXAMPLES.glob("*.cmtz") if "cfield" in f.read_text())
    with pytest.raises(ValueError, match="real-field only"):
        emit_isa(ir(path))


class TestMatPow:
    def _bundle(self, d, eps, n):
        src = (f"field(17, 19);\nroots(17) as M;\n"
               f"matpow(M, {d}, {eps}) as P;\nmeasure(P, mod_p) as out;\n")
        prog = ir(src)
        return prog, emit_isa(prog, matrix_dims={"M": n})

    def test_fibonacci(self):
        prog = ir(EXAMPLES / "fibonacci.cmtz")
        b = emit_isa(prog, matrix_dims={"fib_matrix": 2})
        m = run_image(b.core, b.image, matrices={"fib_matrix": [[1, 1], [1, 0]]})
        # F^12 = [[F13, F12], [F12, F11]] = [[233, 144], [144, 89]] mod 17
        assert m.read_matrix("F12") == [[233 % 17, 144 % 17],
                                        [144 % 17, 89 % 17]]

    @pytest.mark.parametrize("d,eps,n", [
        (1, 0.5, 2), (2, 1.0, 3), (12, 0.5, 2), (37, 1.0, 3),
        (100, 1.0, 2), (65, 0.5, 4), (511, 1.0, 2), (520, 1.0, 1),
    ])
    def test_powers(self, d, eps, n):
        prog, b = self._bundle(d, eps, n)
        node = matpows(prog)[0]
        rng = random.Random(d)
        for _ in range(3):
            mat = [[rng.randrange(17) for _ in range(n)] for _ in range(n)]
            m = run_image(b.core, b.image, matrices={"M": mat},
                          scramble=d)
            assert m.status == S.HALTED
            assert m.read_matrix("P") == \
                PythonBackend(prog.ir_field).eval_matpow_matrix(node, mat)
            assert m.read_matrix("M") == mat        # input not clobbered

    @pytest.mark.parametrize("eps", [1.0, 0.75, 0.5])
    def test_every_degree(self, eps):
        mat = [[3, 1], [16, 5]]
        for d in range(1, 300, 7):
            prog, b = self._bundle(d, eps, 2)
            m = run_image(b.core, b.image, matrices={"M": mat})
            assert m.read_matrix("P") == PythonBackend(
                prog.ir_field).eval_matpow_matrix(matpows(prog)[0], mat), d

    def test_needs_dimension(self):
        prog = ir("field(17, 19);\nroots(17) as M;\nmatpow(M, 3, 0.5) as P;\n")
        with pytest.raises(ValueError, match="matrix_dims"):
            emit_isa(prog)


def test_compile_program_isa_backend():
    r = compile_program((EXAMPLES / "08_ntt_butterfly.cmtz").read_text(),
                        backend="isa")
    assert r.ok, r.errors
    out = r.backend_output
    assert isinstance(out["image"], ProgramImage)
    assert out["words"] == len(out["image"].words) > 0


class TestHandlerVerifier:
    def _img(self, isr: str, borrowed=(2,), scratch=range(5, 10)):
        a = assemble(".entry main\n.handler isr\nmain: WAITI\nHALT\n"
                     "isr:\n" + isr + "\nRETI")
        return ProgramImage(17, 19, a.words, a.entry, a.handler,
                            borrowed=list(borrowed),
                            handler_scratch=list(scratch))

    @pytest.mark.parametrize("name", ["05_interrupt", "09_interrupt_handler"])
    def test_examples_pass(self, name):
        b = emit_isa(ir(EXAMPLES / f"{name}.cmtz"))
        assert b.image.borrowed
        assert verify_handler(b.image) is not None

    def test_reversible_use_of_borrowed_passes(self):
        # borrow R2: add a temporary in, rotate, rotate back, take it out
        rep = verify_handler(self._img("""
            LDI  R5, 4
            ADD  R2, R2, R5
            MULI R2, R2, 3
            MUL  R6, R2, R2
            MULI R2, R2, 6
            SUB  R2, R2, R5
        """))
        assert rep.forms == {2: "1*R2"}      # 3 * 6 = 18 = 1 mod 17

    @pytest.mark.parametrize("isr,msg", [
        ("ADD R2, R2, R5", "leaves the handler"),
        ("LDI R5, 1\nADD R2, R2, R5\nLDI R5, 2\nSUB R2, R2, R5", "leaves"),
        ("MUL R2, R2, R5", "not linear"),
        ("LDI R3, 1", "neither borrowed"),
        ("LIX X1, 1", "index register"),
        ("BZ R5, isr", "not allowed"),
        ("ST R5, [X0]", "not allowed"),
    ])
    def test_violations(self, isr, msg):
        with pytest.raises(IsaCatalyticError, match=msg):
            verify_handler(self._img(isr))

    def test_no_reti(self):
        a = assemble(".handler isr\nHALT\nisr: NOP")
        img = ProgramImage(17, 19, a.words, a.entry, a.handler)
        with pytest.raises(IsaCatalyticError, match="without RETI"):
            verify_handler(img)

    @pytest.mark.parametrize("name", ["05_interrupt", "09_interrupt_handler"])
    def test_handler_registers_disjoint_from_main(self, name):
        b = emit_isa(ir(EXAMPLES / f"{name}.cmtz"))
        main_writes = set()
        for w in b.image.words[:b.image.handler]:
            wr = written_registers(decode(w))
            if wr and wr[0] == "R":
                main_writes.add(wr[1])
        scratch = set(b.image.handler_scratch)
        assert not main_writes & scratch
        assert not set(b.image.borrowed) & scratch


def test_rerun_same_machine():
    """The board reuses one core: second run sees the first run's leftovers."""
    from cmtz.isa import Machine
    for name in ["08_ntt_butterfly", "fibonacci", "13_belief_propagation"]:
        prog = ir(EXAMPLES / f"{name}.cmtz")
        b = emit_isa(prog, matrix_dims={n.matrix_name: 3 for n in matpows(prog)})
        m = Machine(b.core, b.image)
        m.scramble(7)
        rng = random.Random(name)
        p = prog.ir_field.p
        for _ in range(5):
            ins = {n: rng.randrange(p) for n in b.inputs}
            mats = {k: [[rng.randrange(p) for _ in range(n)] for _ in range(n)]
                    for k, n in b.matrices_in.items()}
            m.set_inputs(ins)
            for k, v in mats.items():
                m.write_matrix(k, v)
            assert m.run() == S.HALTED
            exp = evaluate_all(prog, prog.ir_field, ins)
            assert m.outputs() == {o: exp[o] % p for o in b.outputs}
            for node in matpows(prog):
                assert m.read_matrix(node.name) == PythonBackend(
                    prog.ir_field).eval_matpow_matrix(node, mats[node.matrix_name])


class TestCatalyticLoop:
    """An interrupt that borrows a register main is using, mid-loop."""

    def _run(self, img, x, irq=()):
        return run_image(CORE, img, {"x": x}, irq_at=irq, scramble=x)

    def test_restoring_handler_is_invisible_to_main(self):
        from cmtz.isa.demos import catalytic_loop
        img = catalytic_loop()
        verify_handler(img)
        for x in range(17):
            base = self._run(img, x)
            assert base.status == S.HALTED and not base.dispatches
            for k in range(0, base.retired, 3):
                m = self._run(img, x, (k,))
                assert m.status == S.HALTED
                assert m.outputs()["acc"] == base.outputs()["acc"], (x, k)
                assert m.R[2] == x
                assert m.dispatches and m.dispatches[0][0] == k

    def test_non_restoring_handler_rejected_and_visible(self):
        from cmtz.isa.demos import catalytic_loop
        bad = catalytic_loop(restore=False)
        with pytest.raises(IsaCatalyticError, match="R2"):
            verify_handler(bad)
        good = catalytic_loop()
        changed = 0
        for x in range(17):
            base = self._run(good, x)
            for k in range(0, base.retired, 5):
                changed += self._run(bad, x, (k,)).outputs()["acc"] != \
                    base.outputs()["acc"]
        assert changed > 0


def test_verifier_precondition_values_below_p():
    a = assemble(".entry main\n.handler isr\nmain: LDI R2, 18\nWAITI\nHALT\n"
                 "isr: ADD R2, R2, R0\nRETI")
    img = ProgramImage(17, 19, a.words, a.entry, a.handler, borrowed=[2])
    with pytest.raises(IsaCatalyticError, match=r"may leave it in \[p, q\)"):
        verify_handler(img)

"""cmtz-core ISA: encoding, assembler, program image, simulator, library."""

import random
from pathlib import Path

import pytest

from cmtz.field import horner_eval_Fq
from cmtz.isa import (
    AsmError, CoreConfig, Instr, Machine, ProgramImage, SimError, assemble,
    decode, encode, run_image,
)
from cmtz.isa.encoding import FORMATS, OPS
from cmtz.isa import sim as S

CORE = CoreConfig(17, 19)
LIB = (Path(__file__).resolve().parent.parent / "isa" / "lib" / "matrix.s").read_text()


def image(src: str, *more: str, **kw) -> ProgramImage:
    a = assemble(src, *more)
    return ProgramImage(17, 19, a.words, entry=a.entry, handler=a.handler, **kw)


def run(src: str, *more: str, regs=None, data=None, **kw) -> Machine:
    m = Machine(CORE, image(src, *more))
    for r, v in (regs or {}).items():
        m.R[r] = v
    for addr, v in (data or {}).items():
        m.D[addr] = v
    m.run(**kw)
    return m


class TestEncoding:
    def test_roundtrip_every_op(self):
        rng = random.Random(0)
        for op in OPS.values():
            for _ in range(20):
                f = {"a": rng.randrange(16), "b": rng.randrange(16),
                     "c": rng.randrange(16)}
                imm = (rng.randrange(-1024, 1024) if op.fmt == "XXS"
                       else rng.randrange(2048))
                ins = Instr(op.name, **f, imm=imm)
                assert decode(encode(ins)) == ins

    def test_opcodes_unique_and_fit(self):
        codes = [o.code for o in OPS.values()]
        assert len(set(codes)) == len(codes)
        assert max(codes) < 64
        assert all(o.fmt in FORMATS for o in OPS.values())

    def test_range_checks(self):
        with pytest.raises(ValueError):
            encode(Instr("LIX", 1, imm=2048))
        with pytest.raises(ValueError):
            encode(Instr("ADDIX", 1, 1, imm=1024))
        with pytest.raises(ValueError):
            decode(63 << 26)


class TestAssembler:
    def test_labels_and_directives(self):
        a = assemble("""
            .entry start
            .handler isr
            nop_label: NOP
            start:  LIX X1, 3     ; comment
            loop:   ADDIX X1, X1, -1
                    BNEX X1, X0, loop
                    HALT
            isr:    RETI
        """)
        assert a.entry == 1 and a.handler == 5
        assert a.instrs[3] == Instr("BNEX", 1, 0, imm=2)
        assert a.instrs[2].imm == -1

    def test_memory_operand(self):
        a = assemble("LD R3, [X2+5]\nST R4, [X1]")
        assert a.instrs[0] == Instr("LD", 3, 2, imm=5)
        assert a.instrs[1] == Instr("ST", 4, 1, imm=0)

    @pytest.mark.parametrize("bad", [
        "FOO R1", "ADD R1, R2", "ADD R1, R2, R32", "LIX X16, 0",
        "LD R1, X2", "JMP nowhere", "a: NOP\na: NOP", ".bogus 1",
    ])
    def test_errors(self, bad):
        with pytest.raises(AsmError):
            assemble(bad)


class TestImage:
    def test_json_roundtrip(self):
        img = image("LDI R1, 3\nHALT", inputs={"x": 1}, outputs={"y": 1})
        back = ProgramImage.from_json(img.to_json())
        assert back == img and back.crc == img.crc

    def test_crc_detects_tampering(self):
        text = image("HALT").to_json().replace('"entry": 0', '"entry": 1')
        with pytest.raises(ValueError, match="crc"):
            ProgramImage.from_json(text)

    @pytest.mark.parametrize("src,msg", [
        ("JMP 5", "jump target"),
        ("MULI R1, R1, 17\nHALT", "MULI immediate"),
        ("LDI R1, 19\nHALT", "immediate"),
    ])
    def test_validate_rejects(self, src, msg):
        with pytest.raises(ValueError, match=msg):
            image(src).validate(CORE)

    def test_validate_field_and_size(self):
        img = image("HALT")
        with pytest.raises(ValueError, match="field"):
            img.validate(CoreConfig(7, 7))
        big = ProgramImage(17, 19, [0] * 513)
        with pytest.raises(ValueError, match="program memory"):
            big.validate(CORE)


class TestArithmetic:
    """Every field op against Python, exhaustively over F_17."""

    @pytest.mark.parametrize("op,fn", [
        ("ADD", lambda a, b: (a + b) % 17),
        ("SUB", lambda a, b: (a - b) % 17),
        ("MUL", lambda a, b: a * b % 17),
        ("MAC", lambda a, b: (a + a * b) % 17),     # R1 = R1 + R1*R2
    ])
    def test_binary(self, op, fn):
        img = image(f"{op} R1, R1, R2\nHALT")
        for a in range(17):
            for b in range(17):
                m = Machine(CORE, img)
                m.R[1], m.R[2] = a, b
                m.run()
                assert m.R[1] == fn(a, b), (op, a, b)

    def test_muli(self):
        for k in range(17):
            img = image(f"MULI R1, R2, {k}\nHALT")
            for a in range(17):
                m = Machine(CORE, img)
                m.R[2] = a
                m.run()
                assert m.R[1] == a * k % 17

    def test_horner_redp(self):
        rng = random.Random(3)
        for _ in range(50):
            coeffs = [rng.randrange(19) for _ in range(rng.randrange(1, 5))]
            x = rng.randrange(17)
            src = [f"LDI R1, {coeffs[-1]}"]
            src += [f"HORNER R1, R2, {c}" for c in reversed(coeffs[:-1])]
            src += ["REDP R3, R1", "HALT"]
            m = run("\n".join(src), regs={2: x})
            assert m.R[1] == horner_eval_Fq(coeffs, x, 19)
            assert m.R[3] == m.R[1] % 17

    def test_r0_and_x0_are_zero(self):
        m = run("LDI R0, 5\nLIX X0, 5\nADD R1, R0, R0\nHALT")
        assert m.R[0] == 0 and m.X[0] == 0 and m.R[1] == 0

    def test_out_of_range_operand_raises(self):
        with pytest.raises(SimError):
            run("LDI R1, 18\nADD R2, R1, R1\nHALT")

    def test_index_ops_wrap(self):
        m = run("LIX X1, 2047\nMULX X2, X1, X1\nADDIX X3, X0, -1\n"
                "LDI R1, 9\nFTOX X4, R1\nHALT")
        assert m.X[2] == (2047 * 2047) & 0xFFFF
        assert m.X[3] == 0xFFFF and m.X[4] == 9


class TestControl:
    def test_counted_loop_and_memory(self):
        m = run("""
                LIX  X1, 0
                LIX  X2, 10
                LDI  R1, 1
        loop:   ST   R1, [X1+100]
                ADD  R1, R1, R1
                ADDIX X1, X1, 1
                BLTX X1, X2, loop
                HALT
        """)
        assert m.status == S.HALTED
        assert m.D[100:110] == [pow(2, i, 17) for i in range(10)]
        assert m.retired == 3 + 4 * 10 + 1

    def test_branch_on_field_value(self):
        m = run("BZ R1, z\nLDI R2, 1\nHALT\nz: LDI R2, 2\nHALT", regs={1: 0})
        assert m.R[2] == 2
        m = run("BNZ R1, z\nLDI R2, 1\nHALT\nz: LDI R2, 2\nHALT", regs={1: 3})
        assert m.R[2] == 2

    def test_call_ret(self):
        m = run(".entry main\nf: LDI R1, 7\nRET\nmain: CALL f\nHALT")
        assert m.R[1] == 7 and m.status == S.HALTED

    @pytest.mark.parametrize("src,code", [
        ("f: CALL f", S.E_OVERFLOW),
        ("RET", S.E_UNDERFLOW),
        ("RETI", S.E_RETI),
        ("LIX X1, 1024\nLD R1, [X1]\nHALT", S.E_ADDR),
        ("NOP", S.E_PC),
    ])
    def test_errors(self, src, code):
        m = run(src)
        assert m.status == S.ERROR and m.error == code

    def test_step_limit(self):
        m = run("l: JMP l", max_steps=100)
        assert m.status == S.ERROR and m.error == S.E_STEPS


class TestInterrupts:
    SRC = """
        .entry main
        .handler isr
        main:   LDI R1, 1
                LDI R2, 2
                LDI R3, 3
                WAITI
                HALT
        isr:    ADD R10, R10, R0
                LDI R11, 9
                RETI
    """

    def test_no_trigger_waits(self):
        m = run(self.SRC)
        assert m.status == S.RUNNING and m.waiting and m.retired == 3

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_trigger_anywhere_runs_once(self, k):
        m = run(self.SRC, irq_at=(k,))
        assert m.status == S.HALTED and m.R[11] == 9
        assert m.R[1:4] == [1, 2, 3]
        assert m.retired == 3 + 3 + 2      # main, handler, WAITI + HALT

    def test_trigger_during_handler_is_served_after_reti(self):
        m = run(self.SRC, irq_at=(1, 2))   # 2 lands inside the handler
        assert m.status == S.HALTED
        assert m.retired == 3 + 3 + 3 + 2

    def test_no_handler_drops_trigger(self):
        m = run("LDI R1, 1\nHALT", irq_at=(0,))
        assert m.status == S.HALTED and m.retired == 2


class TestMatrixLibrary:
    def _mat(self, rng, n):
        return [[rng.randrange(17) for _ in range(n)] for _ in range(n)]

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_matmul_ident_copy(self, n):
        rng = random.Random(n)
        a, b = self._mat(rng, n), self._mat(rng, n)
        nn = n * n
        src = f"""
            .entry main
            main:   LIX X4, 0
                    LIX X5, {nn}
                    LIX X6, {2 * nn}
                    LIX X7, {n}
                    CALL __matmul
                    LIX X6, {3 * nn}
                    CALL __matident
                    LIX X4, {2 * nn}
                    LIX X6, {4 * nn}
                    CALL __matcopy
                    HALT
        """
        data = {i: v for i, v in enumerate(sum(a, []) + sum(b, []))}
        m = run(src, LIB, data=data)
        want = [[sum(a[i][k] * b[k][j] for k in range(n)) % 17
                 for j in range(n)] for i in range(n)]
        flat = sum(want, [])
        assert m.D[2 * nn:3 * nn] == flat
        assert m.D[3 * nn:4 * nn] == [int(i == j) for i in range(n)
                                      for j in range(n)]
        assert m.D[4 * nn:5 * nn] == flat
        assert m.X[4:8] == [2 * nn, nn, 4 * nn, n]      # args preserved

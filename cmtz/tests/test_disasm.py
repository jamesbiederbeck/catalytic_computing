from cmtz.isa import assemble
from cmtz.isa.disasm import describe
from cmtz.isa.encoding import Instr, encode


def d(op, a=0, b=0, c=0, imm=0):
    return describe(0, encode(Instr(op, a, b, c, imm)))


def test_alu_reads_sources_and_writes_destination():
    x = d("ADD", 3, 4, 5)
    assert (x["reads"], x["writes"], x["mem"], x["target"]) == (["R4", "R5"], ["R3"], None, None)
    assert x["text"] == "ADD R3, R4, R5"


def test_ops_that_accumulate_also_read_their_destination():
    assert d("MAC", 3, 4, 5)["reads"] == ["R3", "R4", "R5"]
    assert d("HORNER", 2, 6, imm=7)["reads"] == ["R2", "R6"]


def test_r0_and_x0_are_neither_read_nor_written():
    x = d("ADD", 0, 0, 5)
    assert x["writes"] == [] and x["reads"] == ["R5"]
    assert d("ADDX", 2, 0, 0)["reads"] == []


def test_index_registers_and_memory():
    assert d("LIX", 4, imm=9)["writes"] == ["X4"]
    ld = d("LD", 3, 2, imm=5)
    assert (ld["reads"], ld["writes"], ld["mem"]) == (["X2"], ["R3"], "read")
    st = d("ST", 3, 2, imm=5)
    assert (st["reads"], st["writes"], st["mem"]) == (["R3", "X2"], [], "write")


def test_branches_read_and_name_their_target():
    b = d("BNEX", 1, 2, imm=17)
    assert (b["reads"], b["writes"], b["target"]) == (["X1", "X2"], [], 17)
    assert d("BZ", 5, imm=3)["reads"] == ["R5"]
    assert d("CALL", imm=40)["target"] == 40
    assert d("HALT")["reads"] == [] and d("HALT")["target"] is None


def test_a_whole_image():
    a = assemble("LDI R1, 5\nADD R2, R1, R1\nHALT\n")
    rows = [describe(i, w) for i, w in enumerate(a.words)]
    assert [r["op"] for r in rows] == ["LDI", "ADD", "HALT"]
    assert rows[1]["reads"] == ["R1"] and rows[1]["writes"] == ["R2"]


def test_raw_operand_fields_are_included():
    x = d("LD", 3, 2, imm=5)
    assert (x["a"], x["b"], x["imm"]) == (3, 2, 5)

"""Soundness of the handler verifier (analysis/isa_catalytic.py): values
outside F_p, registers the handler reads, scratch disjointness. The
failures it now rejects are shown on the RTL, since the simulator refuses
out-of-range operands instead of modelling what the hardware does."""

import shutil
import subprocess

import pytest

from cmtz.analysis.isa_catalytic import (
    IsaCatalyticError, noncanonical_registers, verify_handler,
)
from cmtz.backends.isa_core import emit_core
from cmtz.isa import CoreConfig, ProgramImage, assemble, decode
from cmtz.isa import projection as PR
from cmtz.isa.demos import catalytic_loop, noncanonical_handler
from cmtz.isa.host import CoreLink

CORE = CoreConfig(17, 19)
IVERILOG = shutil.which("iverilog") and shutil.which("vvp")


def img(src, **kw):
    a = assemble(src)
    return ProgramImage(17, 19, a.words, a.entry, a.handler, **kw)


def rejects(image, match):
    with pytest.raises(IsaCatalyticError, match=match):
        verify_handler(image)


MAIN = ".entry main\n.handler isr\nmain: {main}\nWAITI\nHALT\nisr: {isr}\nRETI"


class TestNoncanonicalValues:
    def test_handler_reading_a_register_main_sets_to_18(self):
        rejects(noncanonical_handler(), r"R7 is read by the handler .* \[p, q\)")

    @pytest.mark.parametrize("main", [
        "LIX X1, 0\nLD R2, [X1]",                   # data memory holds < q
        "LDI R3, 2\nHORNER R3, R3, 14\nMOV R2, R3",  # 2*2 + 14 = 18, via MOV
        "LDI R4, 17\nMOV R5, R4\nMOV R2, R5",        # MOV chain
    ])
    def test_borrowed_register_reachable_outside_the_field(self, main):
        rejects(img(MAIN.format(main=main, isr="ADD R2, R2, R0"),
                    inputs={"x": 2}, borrowed=[2]),
                r"R2 is borrowed .* \[p, q\)")

    def test_redp_makes_a_value_canonical(self):
        verify_handler(img(MAIN.format(
            main="LDI R3, 2\nHORNER R3, R3, 14\nREDP R2, R3",
            isr="ADD R2, R2, R0"), inputs={"x": 2}, borrowed=[2]))

    def test_fixpoint(self):
        code = [decode(w) for w in img("LDI R1, 18\nMOV R2, R1\nMOV R3, R2\n"
                                       "ADD R4, R3, R0\nLDI R5, 3\nHALT").words]
        assert noncanonical_registers(code, range(0), 17) == {1, 2, 3}


class TestHandlerReads:
    def test_reading_a_register_the_host_does_not_write(self):
        """A dispatch at the first boundary sees R7's leftover."""
        rejects(img(MAIN.format(main="LDI R7, 3",
                                isr="SUB R2, R2, R7\nADD R2, R2, R7"),
                    inputs={"x": 2}, borrowed=[2]),
                "reads R7, which the host does not write")

    def test_host_written_handler_input_is_fine(self):
        rep = verify_handler(img(MAIN.format(main="NOP",
                                             isr="SUB R2, R2, R7\nADD R2, R2, R7"),
                                 inputs={"x": 2, "k": 7}, borrowed=[2]))
        assert rep.requires_canonical == [2, 7]

    def test_written_before_read_is_fine(self):
        verify_handler(img(MAIN.format(main="NOP",
                                       isr="LDI R7, 4\nSUB R2, R2, R7\nADD R2, R2, R7"),
                           inputs={"x": 2}, borrowed=[2], handler_scratch=[7]))


def test_main_must_not_touch_handler_scratch():
    rejects(img(MAIN.format(main="LDI R5, 1\nADD R3, R5, R5",
                            isr="LDI R5, 4\nADD R2, R2, R5\nSUB R2, R2, R5"),
                inputs={"x": 2}, borrowed=[2], handler_scratch=[5]),
            "uses R5, which is handler scratch")


def test_existing_handlers_still_verify():
    assert verify_handler(catalytic_loop()).requires_canonical == [2]
    assert verify_handler(PR.gate_irq(1, 1).image).requires_canonical == [7]


def test_host_refuses_inputs_outside_the_field():
    link = CoreLink(port=None, core=CORE)
    with pytest.raises(ValueError, match="t1=18"):
        link.run(PR.gate_clean().image, {"t1": 18}, load=False)


# ── on the RTL ───────────────────────────────────────────────────────────

def _raw_run(tmp_path, image: ProgramImage, regs: dict[int, int],
             irq_retired: int | None, show: list[int]) -> dict[int, int]:
    """Run `image` on the generated core with raw register writes (any
    value below q) and an optional interrupt at a retired count; return
    the registers in `show`."""
    words = "\n".join(f"    pmw({a}, 32'h{w:08x});" for a, w in enumerate(image.words))
    writes = "\n".join(f"    rw_({r}, {v});" for r, v in regs.items())
    shows = "\n".join(f'    $display("R{r}=%0d", dut.R[{r}]);' for r in show)
    irq = irq_retired if irq_retired is not None else -1
    tb = f"""`timescale 1ns/1ps
module raw_tb;
reg clk = 0; always #5 clk = ~clk;
reg start = 0, irq = 0, pm_we = 0, r_we = 0;
reg [8:0] pm_addr = 0; reg [31:0] pm_wdata = 0;
reg [4:0] r_addr = 0; reg [4:0] r_wdata = 0;
wire [1:0] status; wire [31:0] retired; wire boundary;
cmtz_core dut (.clk(clk), .start(start), .abort(1'b0), .irq(irq),
    .cfg_len(10'd{len(image.words)}), .cfg_entry(10'd{image.entry}),
    .cfg_handler(10'd{image.handler or 0}), .cfg_has_handler(1'b{int(image.handler is not None)}),
    .pm_we(pm_we), .pm_addr(pm_addr), .pm_wdata(pm_wdata),
    .dm_we(1'b0), .dm_addr(10'd0), .dm_wdata(5'd0), .dm_rdata(),
    .r_we(r_we), .r_addr(r_addr), .r_wdata(r_wdata), .r_rdata(),
    .x_addr(4'd0), .x_rdata(), .status(status), .error(), .retired(retired),
    .disp_retired(), .disp_pc(), .disp_count(), .boundary(boundary), .pc_out());
task pmw(input integer a, input [31:0] w);
    begin pm_we = 1; pm_addr = a; pm_wdata = w; @(negedge clk); pm_we = 0; end
endtask
task rw_(input integer r, input integer v);
    begin r_we = 1; r_addr = r; r_wdata = v; @(negedge clk); r_we = 0; end
endtask
integer fired = 0, cyc = 0;
initial begin
    repeat (2) @(negedge clk);
{words}
{writes}
    start = 1; @(negedge clk); start = 0;
    while (status == 2'd1 && cyc < 1000000) begin
        irq = (!fired && boundary && retired == {irq});
        if (irq) fired = 1;
        @(negedge clk); cyc = cyc + 1;
    end
{shows}
    $finish;
end
endmodule
"""
    (tmp_path / "cmtz_core.v").write_text(emit_core(CORE))
    (tmp_path / "raw_tb.v").write_text(tb)
    subprocess.run(["iverilog", "-g2005", "-o", "s.out", "cmtz_core.v", "raw_tb.v"],
                   cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run(["vvp", "s.out"], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    vals = dict(line.split("=") for line in out.split() if line.startswith("R"))
    return {int(k[1:]): int(v) for k, v in vals.items()}


@pytest.mark.skipif(not IVERILOG, reason="iverilog/vvp not installed")
class TestRtlCounterexamples:
    def test_handler_reading_18_corrupts_the_borrowed_register(self, tmp_path):
        """What the verifier used to accept: x = 0 comes back as 13."""
        image = noncanonical_handler()
        assert _raw_run(tmp_path, image, {2: 0}, None, [2]) == {2: 0}
        assert _raw_run(tmp_path, image, {2: 0}, 5, [2]) == {2: 13}

    @pytest.mark.parametrize("t1,back", [(17, 0), (18, 1)])
    def test_mask_outside_the_field_is_not_restored(self, tmp_path, t1, back):
        """gate_clean is proven for masks in [0, p); a mask register holding
        17 or 18 is reduced by the first ADD and comes back as 0 or 1 (the
        output is still right, since 17 = 0 and 18 = 1 mod 17)."""
        demo = PR.gate_clean()
        vals = {**PR.gate_inputs(PR.GATES["AND"], 1, 1), "t2": 5, "y": 3}
        regs = {demo.image.inputs[k]: v for k, v in vals.items()}
        regs[PR.T1] = t1
        got = _raw_run(tmp_path, demo.image, regs, None, [PR.T1, PR.T2, PR.Y])
        assert got == {PR.T1: back, PR.T2: 5, PR.Y: 4}

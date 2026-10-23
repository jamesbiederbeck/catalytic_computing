"""Behavioural simulation model of the Gowin SDPX9B primitive (18-bit words only).

Used only when the core hand-instantiates the block RAM (``bank_prim="sdp"``).  Guarded by
``ifndef SYNTHESIZE``: synthesis (yosys / apio define SYNTHESIZE) uses the real primitive.
Read timing follows the Gowin definition: the address is registered by the block, the
bypass read is combinational from that register, and READ_MODE=1 adds an output register.
Write enable is CEA; byte enables (ADA[1:0]) are assumed all set.
"""

MODEL = """\
// Simulation model of the Gowin SDPX9B primitive (18-bit words, byte enables assumed set).
`ifndef SYNTHESIZE
module SDPX9B #(
    parameter READ_MODE = 1'b0,
    parameter BIT_WIDTH_0 = 36,
    parameter BIT_WIDTH_1 = 36,
    parameter [2:0] BLK_SEL_0 = 3'b000,
    parameter [2:0] BLK_SEL_1 = 3'b000,
    parameter RESET_MODE = "SYNC"
) (
    input  wire        CLKA, CEA, CLKB, CEB, OCE, RESETA, RESETB,
    input  wire [13:0] ADA, ADB,
    input  wire [2:0]  BLKSELA, BLKSELB,
    input  wire [35:0] DI,
    output wire [35:0] DO
);
    reg [17:0] mem [0:1023];
    reg [9:0]  adb_q = 10'd0;
    reg [17:0] dout_q = 18'd0;
    integer i;
    initial for (i = 0; i < 1024; i = i + 1) mem[i] = 18'd0;
    always @(posedge CLKA) if (CEA) mem[ADA[13:4]] <= DI[17:0];
    always @(posedge CLKB) if (CEB) adb_q <= ADB[13:4];
    wire [17:0] bypass = mem[adb_q];
    always @(posedge CLKB) if (OCE) dout_q <= bypass;
    assign DO = {18'd0, READ_MODE ? dout_q : bypass};
endmodule
`endif
"""

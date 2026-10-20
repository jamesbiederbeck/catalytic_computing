// UART front end for the generated `cmtz_mul` datapath (F_17 multiply).
//
// Protocol, 115200 8N1: send two bytes a, b (each 0..16). The reply is one
// byte, a*b mod 17. If a or b is 17 or more, the reply is 0xFF. Wait for the
// reply before sending the next pair; bytes sent while a reply is going out
// are ignored.
//
// LEDs (active low): leds[4:0] show the last reply, leds[5] lights after an
// error reply.

module uart_rx #(
    parameter CLKS_PER_BIT = 234            // 27 MHz / 115200
) (
    input  wire       clk,
    input  wire       rx,
    output reg  [7:0] data   = 0,
    output reg        strobe = 0            // one clock, byte received with good stop bit
);
    reg rx_m = 1'b1, rx_s = 1'b1;           // 2-flop synchronizer
    always @(posedge clk) begin rx_m <= rx; rx_s <= rx_m; end

    localparam IDLE = 0, START = 1, DATA = 2, STOP = 3;
    reg [1:0]  st  = IDLE;
    reg [15:0] cnt = 0;
    reg [2:0]  bi  = 0;

    always @(posedge clk) begin
        strobe <= 1'b0;
        case (st)
            IDLE: if (!rx_s) begin cnt <= 0; st <= START; end
            START: if (cnt == CLKS_PER_BIT / 2 - 1) begin   // middle of start bit
                if (!rx_s) begin cnt <= 0; bi <= 0; st <= DATA; end
                else st <= IDLE;
            end else cnt <= cnt + 1;
            DATA: if (cnt == CLKS_PER_BIT - 1) begin        // middle of each data bit
                cnt  <= 0;
                data <= {rx_s, data[7:1]};                  // LSB first
                bi   <= bi + 1;
                if (bi == 7) st <= STOP;
            end else cnt <= cnt + 1;
            STOP: if (cnt == CLKS_PER_BIT - 1) begin
                strobe <= rx_s;
                st <= IDLE;
            end else cnt <= cnt + 1;
        endcase
    end
endmodule

module uart_tx #(
    parameter CLKS_PER_BIT = 234
) (
    input  wire       clk,
    input  wire [7:0] data,
    input  wire       start,                // one clock, ignored while busy
    output reg        tx   = 1'b1,
    output wire       busy
);
    reg [9:0]  sh     = 10'h3ff;            // {stop 1, byte, start 0}
    reg [15:0] cnt    = 0;
    reg [3:0]  bi     = 0;
    reg        active = 1'b0;
    assign busy = active;

    always @(posedge clk) begin
        if (!active) begin
            tx <= 1'b1;
            if (start) begin sh <= {1'b1, data, 1'b0}; cnt <= 0; bi <= 0; active <= 1'b1; end
        end else begin
            tx <= sh[0];
            if (cnt == CLKS_PER_BIT - 1) begin
                cnt <= 0;
                sh  <= {1'b1, sh[9:1]};
                bi  <= bi + 1;
                if (bi == 9) active <= 1'b0;
            end else cnt <= cnt + 1;
        end
    end
endmodule

module cmtz_mul_uart #(
    parameter CLKS_PER_BIT = 234
) (
    input  wire       clk,
    input  wire       rx,
    output wire       tx,
    output wire [5:0] leds
);
    wire [7:0] rx_data;
    wire       rx_strobe;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rx(rx), .data(rx_data), .strobe(rx_strobe));

    reg  [7:0] reply    = 0;
    reg        tx_start = 1'b0;
    wire       tx_busy;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .data(reply), .start(tx_start), .tx(tx), .busy(tx_busy));

    reg  [7:0] a = 0, b = 0;
    reg        in_valid = 1'b0;
    wire       out_valid;
    wire [4:0] product;
    cmtz_mul dut (
        .clk(clk), .in_valid(in_valid),
        .in_embed_0(a[4:0]), .in_embed_1(b[4:0]),
        .out_product(product), .out_valid(out_valid));

    localparam S_A = 0, S_B = 1, S_CHK = 2, S_RUN = 3, S_SEND = 4, S_WAIT = 5, S_DONE = 6;
    reg [2:0] state = S_A;
    reg       err   = 1'b0;

    always @(posedge clk) begin
        in_valid <= 1'b0;
        tx_start <= 1'b0;
        case (state)
            S_A: if (rx_strobe) begin a <= rx_data; state <= S_B; end
            S_B: if (rx_strobe) begin b <= rx_data; state <= S_CHK; end
            S_CHK: if (a >= 17 || b >= 17) begin
                reply <= 8'hFF; err <= 1'b1; state <= S_SEND;
            end else begin
                in_valid <= 1'b1; err <= 1'b0; state <= S_RUN;
            end
            S_RUN: if (out_valid) begin reply <= {3'b000, product}; state <= S_SEND; end
            S_SEND: if (!tx_busy) begin tx_start <= 1'b1; state <= S_WAIT; end
            S_WAIT: state <= S_DONE;                   // let tx_busy rise
            S_DONE: if (!tx_busy) state <= S_A;
            default: state <= S_A;
        endcase
    end

    assign leds = ~{err, reply[4:0]};                  // LEDs are active low
endmodule

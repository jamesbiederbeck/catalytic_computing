// UART receiver and transmitter used by verilog_uart.py (8N1, mid-bit sampling).
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


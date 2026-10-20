`timescale 1ns/1ps
// Sends every pair a, b in 0..16 through the UART front end and checks the
// reply against (a*b) % 17, then checks the 0xFF reply for out-of-range input.
// Uses 8 clocks per bit so it runs fast. Prints PASS or FAIL.
module cmtz_mul_uart_tb;
    localparam CPB = 8;
    reg clk = 1'b0;
    reg rx  = 1'b1;
    wire tx;
    wire [5:0] leds;
    cmtz_mul_uart #(.CLKS_PER_BIT(CPB)) dut (.clk(clk), .rx(rx), .tx(tx), .leds(leds));
    always #5 clk = ~clk;

    integer errors = 0;
    integer checked = 0;

    task send_byte;
        input [7:0] v;
        integer i;
        begin
            rx = 1'b0; repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                rx = v[i]; repeat (CPB) @(posedge clk);
            end
            rx = 1'b1; repeat (CPB) @(posedge clk);
        end
    endtask

    task recv_byte;
        output [7:0] v;
        integer i;
        begin
            @(negedge tx);                          // start bit
            repeat (CPB / 2) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                repeat (CPB) @(posedge clk); v[i] = tx;
            end
            repeat (CPB) @(posedge clk);            // stop bit
        end
    endtask

    task check;
        input [7:0] a;
        input [7:0] b;
        reg [7:0] got, want;
        begin
            want = (a >= 17 || b >= 17) ? 8'hFF : ((a * b) % 17);
            send_byte(a); send_byte(b);
            recv_byte(got);
            checked = checked + 1;
            if (got !== want) begin
                $display("FAIL: %0d * %0d -> %0d, expected %0d", a, b, got, want);
                errors = errors + 1;
            end
        end
    endtask

    integer x, y;
    initial begin
        repeat (20) @(posedge clk);
        for (x = 0; x < 17; x = x + 1)
            for (y = 0; y < 17; y = y + 1)
                check(x, y);
        check(17, 1); check(1, 200); check(255, 255);
        check(3, 5);                                // still works after an error
        if (errors == 0) $display("PASS");
        else $display("FAIL: %0d errors in %0d checks", errors, checked);
        $finish;
    end

    initial begin
        #20_000_000;
        $display("FAIL: timeout after %0d checks", checked);
        $finish;
    end
endmodule

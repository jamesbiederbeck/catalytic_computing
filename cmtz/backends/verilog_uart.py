"""UART front end for a Verilog datapath bundle (Tang Nano 20K style board).

Wraps the module from `emit_verilog` so a host can drive it over a serial
port with no extra hardware, and generates a whole-design testbench that
bit-bangs the same protocol.

Protocol, 8N1:
  request  one byte per input lane, in port order. A complex port is two
           lanes, re then im. Every lane is a value in 0..p-1.
  reply    one byte per output lane, in port order, once the datapath
           result is ready.
  error    if any request lane is >= p, every reply byte is 0xFF.
           (p must be <= 255 so 0xFF is never a legal lane value.)
  resync   if a request stops partway for TIMEOUT_CLKS, the bytes received
           so far are dropped. Bytes sent while a reply is going out are
           ignored, so a host should read the whole reply first.
LEDs (active low): leds[4:0] show the low bits of the first reply byte,
leds[5] lights after an error reply.

Debug protocol (bundle emitted with debug=True) puts a command byte in front,
so a host can read the design back:
  'I' (0x49)  info frame, 14 bytes: "CMTZ", 1, NI, NO, ND, p, latency, then
              the program hash (crc32, big endian). The hash lets a host check
              the bitstream was built from the program it compiled.
  'R' (0x52)  run: then the request as above; the reply is as above.
  'S' (0x53)  state: ND bytes, one per lane of every input and registered node
              (the `nodes` order of the bundle, re then im). Valid once the
              last request has settled, `latency` clocks after it was
              accepted; the datapath keeps running on the held inputs.
  other       ignored. A command that stalls for TIMEOUT_CLKS is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .verilog_backend import VerilogBundle



@dataclass
class UartWrapper:
    source: str                 # uart_rx, uart_tx and the wrapper top
    testbench: str
    top_name: str
    request_bytes: int
    reply_bytes: int


def emit_uart_wrapper(bundle: VerilogBundle,
                      top_name: str | None = None) -> UartWrapper:
    """Emit a UART wrapper and testbench around `bundle`'s datapath."""
    p = bundle.field_p
    if p > 255:
        raise ValueError(f"p={p} does not fit one byte per lane")
    pw = max(1, (p - 1).bit_length())
    top = top_name or f"{bundle.module_name}_uart"
    ins = [pt for pt in bundle.ports if pt.direction == "input"]
    outs = [pt for pt in bundle.ports if pt.direction == "output"]
    if not ins or not outs:
        raise ValueError("UART wrapper needs at least one input and one output "
                         "port (use inputs= when emitting the datapath)")

    # request lane k -> input port slices
    ni = no = 0
    in_wires, dut_conns = [], []
    for pt in ins:
        if pt.is_complex:
            expr = f"{{req[{ni + 1}][{pw - 1}:0], req[{ni}][{pw - 1}:0]}}"
            ni += 2
        else:
            expr = f"req[{ni}][{pw - 1}:0]"
            ni += 1
        in_wires.append(f"    wire [{pt.width - 1}:0] {pt.name}_w = {expr};")
        dut_conns.append(f".{pt.name}({pt.name}_w)")
    out_wires, load = [], []
    for pt in outs:
        out_wires.append(f"    wire [{pt.width - 1}:0] {pt.name}_w;")
        dut_conns.append(f".{pt.name}({pt.name}_w)")
        lanes = ([f"{pt.name}_w[{pw - 1}:0]", f"{pt.name}_w[{2 * pw - 1}:{pw}]"]
                 if pt.is_complex else [f"{pt.name}_w"])
        for e in lanes:
            load.append(f"rep[{no}] <= {e};")
            no += 1
    dut_conns += [".clk(clk)", ".in_valid(in_valid)", ".out_valid(out_valid)"]
    if ni > 255 or no > 255:
        raise ValueError("too many lanes for the byte counter")

    if bundle.debug:
        return _emit_debug(bundle, top, ni, no, pw, in_wires, out_wires,
                           dut_conns, load, err_load_lines=[
                               f"rep[{k}] <= 8'hFF;" for k in range(no)])

    sig_in = "; ".join(f"{pt.name}{' (re, im)' if pt.is_complex else ''}"
                       for pt in ins)
    sig_out = "; ".join(f"{pt.name}{' (re, im)' if pt.is_complex else ''}"
                        for pt in outs)
    err_load = " ".join(f"rep[{k}] <= 8'hFF;" for k in range(no))
    nl = "\n"
    top_src = f'''// UART front end for the generated `{bundle.module_name}` datapath, F_{p}.
//
// Request ({ni} byte{'s' * (ni != 1)}, 8N1, each 0..{p - 1}): {sig_in}
// Reply   ({no} byte{'s' * (no != 1)}): {sig_out}
// Any request byte >= {p} makes every reply byte 0xFF. A request that stalls
// partway is dropped after TIMEOUT_CLKS; bytes sent during a reply are
// ignored. LEDs (active low): leds[4:0] low bits of the first reply byte,
// leds[5] lights after an error reply.
module {top} #(
    parameter CLKS_PER_BIT = 234,           // 27 MHz / 115200
    parameter TIMEOUT_CLKS = 6_750_000      // 250 ms at 27 MHz
) (
    input  wire       clk,
    input  wire       rx,
    output wire       tx,
    output wire [5:0] leds
);
    localparam NI = {ni}, NO = {no};

    wire [7:0] rx_data;
    wire       rx_strobe;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rx(rx), .data(rx_data), .strobe(rx_strobe));

    reg  [7:0] req [0:NI-1];
    reg  [7:0] rep [0:NO-1];
    reg  [7:0] cnt      = 0;
    reg        bad      = 1'b0;
    reg        err      = 1'b0;
    reg        tx_start = 1'b0;
    wire       tx_busy;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .data(rep[cnt]), .start(tx_start), .tx(tx), .busy(tx_busy));

    reg        in_valid = 1'b0;
    wire       out_valid;
{nl.join(in_wires)}
{nl.join(out_wires)}
    {bundle.module_name} dut ({", ".join(dut_conns)});

    localparam S_RECV = 0, S_CHK = 1, S_RUN = 2, S_SEND = 3, S_WAIT = 4, S_DONE = 5;
    reg [2:0]  state = S_RECV;
    reg [31:0] idle  = 0;

    always @(posedge clk) begin
        in_valid <= 1'b0;
        tx_start <= 1'b0;
        case (state)
            S_RECV: if (rx_strobe) begin
                req[cnt] <= rx_data;
                if (rx_data >= {_lit8(p)}) bad <= 1'b1;
                if (cnt == NI - 1) begin cnt <= 0; state <= S_CHK; end
                else cnt <= cnt + 1;
            end
            S_CHK: if (bad) begin
                {err_load}
                err <= 1'b1; state <= S_SEND;
            end else begin
                in_valid <= 1'b1; err <= 1'b0; state <= S_RUN;
            end
            S_RUN: if (out_valid) begin
                {" ".join(load)}
                state <= S_SEND;
            end
            S_SEND: if (!tx_busy) begin tx_start <= 1'b1; state <= S_WAIT; end
            S_WAIT: state <= S_DONE;                   // let tx_busy rise
            S_DONE: if (!tx_busy) begin
                if (cnt == NO - 1) begin cnt <= 0; bad <= 1'b0; state <= S_RECV; end
                else begin cnt <= cnt + 1; state <= S_SEND; end
            end
            default: state <= S_RECV;
        endcase

        // resync: drop a half-received request after TIMEOUT_CLKS of silence
        if (state == S_RECV && cnt != 0 && !rx_strobe) begin
            if (idle == TIMEOUT_CLKS - 1) begin idle <= 0; cnt <= 0; bad <= 1'b0; end
            else idle <= idle + 1;
        end else idle <= 0;
    end

    assign leds = ~{{err, rep[0][4:0]}};                // LEDs are active low
endmodule
'''
    uart = Path(__file__).with_name("verilog_uart_modules.v").read_text()
    source = uart + "\n" + top_src

    # ── testbench ──
    nv = len(bundle.vectors)
    init = []
    for v, (il, ol) in enumerate(bundle.vectors):
        for k, x in enumerate(il):
            init.append(f"        vin[{v * ni + k}] = 8'd{x};")
        for k, x in enumerate(ol):
            init.append(f"        vout[{v * no + k}] = 8'd{x};")
    resync = ""
    if ni >= 2:
        resync = f'''        // resync: a lone first byte, then silence longer than the timeout
        send_byte(vin[0]);
        repeat (TO + 100) @(posedge clk);
        check_vec(0);
'''
    tb = f'''`timescale 1ns/1ps
// Whole-design test for {top}: bit-bangs the UART protocol, sends every
// vector, and compares each reply byte with the compiler's Python reference
// ({nv} vector{'s' * (nv != 1)}). Also checks the out-of-range reply and the
// resync timeout. Uses 8 clocks per bit so it runs fast. Prints PASS or FAIL.
module {top}_tb;
    localparam CPB = 8, TO = 2000;
    localparam NI = {ni}, NO = {no}, NV = {nv};
    reg clk = 1'b0;
    reg rx  = 1'b1;
    wire tx;
    wire [5:0] leds;
    {top} #(.CLKS_PER_BIT(CPB), .TIMEOUT_CLKS(TO)) dut (.clk(clk), .rx(rx), .tx(tx), .leds(leds));
    always #5 clk = ~clk;

    reg [7:0] vin  [0:NV*NI-1];
    reg [7:0] vout [0:NV*NO-1];
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

    task check_vec;
        input integer idx;
        integer j;
        reg [7:0] got;
        begin
            for (j = 0; j < NI; j = j + 1) send_byte(vin[idx * NI + j]);
            for (j = 0; j < NO; j = j + 1) begin
                recv_byte(got);
                if (got !== vout[idx * NO + j]) begin
                    $display("FAIL: vector %0d reply byte %0d = %0d, expected %0d",
                             idx, j, got, vout[idx * NO + j]);
                    errors = errors + 1;
                end
            end
            checked = checked + 1;
        end
    endtask

    task check_error;
        integer j;
        reg [7:0] got;
        begin
            send_byte({p});                          // out of range
            for (j = 1; j < NI; j = j + 1) send_byte(0);
            for (j = 0; j < NO; j = j + 1) begin
                recv_byte(got);
                if (got !== 8'hFF) begin
                    $display("FAIL: error reply byte %0d = %0d, expected 255", j, got);
                    errors = errors + 1;
                end
            end
            checked = checked + 1;
        end
    endtask

    integer v;
    initial begin
{nl.join(init)}
        repeat (20) @(posedge clk);
        for (v = 0; v < NV; v = v + 1) check_vec(v);
        check_error;
        check_vec(0);                               // still works after an error
{resync}        check_vec(NV - 1);
        if (errors == 0) $display("PASS");
        else $display("FAIL: %0d errors in %0d checks", errors, checked);
        $finish;
    end

    initial begin
        #200_000_000;
        $display("FAIL: timeout after %0d checks", checked);
        $finish;
    end
endmodule
'''
    return UartWrapper(source=source, testbench=tb, top_name=top,
                       request_bytes=ni, reply_bytes=no)


def _lit8(v: int) -> str:
    return f"8'd{v}"


INFO_LEN = 14
CMD_RUN, CMD_STATE, CMD_INFO = 0x52, 0x53, 0x49


def info_frame(bundle: VerilogBundle, ni: int, no: int, nd: int) -> list[int]:
    h = bundle.program_hash
    return [ord(c) for c in "CMTZ"] + [
        1, ni, no, nd, bundle.field_p, bundle.latency,
        (h >> 24) & 255, (h >> 16) & 255, (h >> 8) & 255, h & 255]


def dbg_layout(bundle: VerilogBundle) -> list[tuple[str, str, int]]:
    """(node, lane, bit offset) for each state byte, in dump order."""
    pw = max(1, (bundle.field_p - 1).bit_length())
    out = []
    for n in bundle.nodes:
        if n.dbg_offset is None:
            continue
        if n.is_complex:
            out += [(n.name, "re", n.dbg_offset),
                    (n.name, "im", n.dbg_offset + pw)]
        else:
            out.append((n.name, "", n.dbg_offset))
    return out


def lane_table(bundle: VerilogBundle) -> dict:
    """Lane layout of the UART protocol, for hosts."""
    inputs, outputs = [], []
    for pt in bundle.ports:
        tgt = inputs if pt.direction == "input" else outputs
        for lane in (("re", "im") if pt.is_complex else ("",)):
            tgt.append({"node": pt.node, "lane": lane, "port": pt.name})
    state = [{"node": n, "lane": lane}
             for n, lane, _ in dbg_layout(bundle)] if bundle.debug else []
    return {"inputs": inputs, "outputs": outputs, "state": state}


def _emit_debug(bundle, top, ni, no, pw, in_wires, out_wires, dut_conns,
                load, err_load_lines) -> UartWrapper:
    p = bundle.field_p
    lanes = dbg_layout(bundle)
    nd = len(lanes)
    if nd > 255:
        raise ValueError("too many state lanes for the byte counter")
    info = info_frame(bundle, ni, no, nd)
    nl = "\n"
    dbg_wires = [f"    wire [7:0] dbgb_{k} = dbg_state_w[{off + pw - 1}:{off}];"
                 for k, (_, _, off) in enumerate(lanes)]
    dbg_cases = nl.join(f"            8'd{k}: tx_byte = dbgb_{k};"
                        for k in range(nd))
    info_cases = nl.join(f"            8'd{k}: tx_byte = 8'd{b};"
                         for k, b in enumerate(info))
    err_load = " ".join(err_load_lines)
    dut_conns = dut_conns + [".dbg_state(dbg_state_w)"]
    sig_in = "; ".join(f"{pt.name}" for pt in bundle.ports
                       if pt.direction == "input")
    top_src = f'''// UART front end with state readback for the generated `{bundle.module_name}`
// datapath, F_{p}. Commands (see cmtz/backends/verilog_uart.py):
//   'I' info frame ({INFO_LEN} bytes), 'R' + {ni} request byte{'s' * (ni != 1)} -> {no} reply
//   byte{'s' * (no != 1)}, 'S' -> {nd} state bytes. Request lanes: {sig_in}.
// Any request byte >= {p} makes every reply byte 0xFF. A command that stalls
// is dropped after TIMEOUT_CLKS. LEDs (active low): leds[4:0] low bits of the
// first reply byte, leds[5] lights after an error reply.
module {top} #(
    parameter CLKS_PER_BIT = 234,           // 27 MHz / 115200
    parameter TIMEOUT_CLKS = 6_750_000      // 250 ms at 27 MHz
) (
    input  wire       clk,
    input  wire       rx,
    output wire       tx,
    output wire [5:0] leds
);
    localparam NI = {ni}, NO = {no}, ND = {nd}, NINFO = {INFO_LEN};

    wire [7:0] rx_data;
    wire       rx_strobe;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rx(rx), .data(rx_data), .strobe(rx_strobe));

    reg  [7:0] req [0:NI-1];
    reg  [7:0] rep [0:NO-1];
    reg  [7:0] cnt      = 0;
    reg  [1:0] mode     = 0;                // 0 reply, 1 state, 2 info
    reg        bad      = 1'b0;
    reg        err      = 1'b0;
    reg        tx_start = 1'b0;
    reg  [7:0] tx_byte;
    wire       tx_busy;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .data(tx_byte), .start(tx_start), .tx(tx), .busy(tx_busy));

    reg        in_valid = 1'b0;
    wire       out_valid;
    wire [{bundle.dbg_width - 1}:0] dbg_state_w;
{nl.join(in_wires)}
{nl.join(out_wires)}
{nl.join(dbg_wires)}
    {bundle.module_name} dut ({", ".join(dut_conns)});

    always @* begin
        case (mode)
        2'd1: case (cnt)
{dbg_cases}
            default: tx_byte = 8'h00;
        endcase
        2'd2: case (cnt)
{info_cases}
            default: tx_byte = 8'h00;
        endcase
        default: tx_byte = rep[cnt];
        endcase
    end
    wire [7:0] last = (mode == 2'd1) ? ND - 1 : (mode == 2'd2) ? NINFO - 1 : NO - 1;

    localparam S_CMD = 0, S_RECV = 1, S_CHK = 2, S_RUN = 3, S_SEND = 4, S_WAIT = 5, S_DONE = 6;
    reg [2:0]  state = S_CMD;
    reg [31:0] idle  = 0;

    always @(posedge clk) begin
        in_valid <= 1'b0;
        tx_start <= 1'b0;
        case (state)
            S_CMD: if (rx_strobe) begin
                if (rx_data == 8'd{CMD_RUN}) begin cnt <= 0; bad <= 1'b0; state <= S_RECV; end
                else if (rx_data == 8'd{CMD_STATE}) begin mode <= 2'd1; cnt <= 0; state <= S_SEND; end
                else if (rx_data == 8'd{CMD_INFO}) begin mode <= 2'd2; cnt <= 0; state <= S_SEND; end
            end
            S_RECV: if (rx_strobe) begin
                req[cnt] <= rx_data;
                if (rx_data >= {_lit8(p)}) bad <= 1'b1;
                if (cnt == NI - 1) begin cnt <= 0; state <= S_CHK; end
                else cnt <= cnt + 1;
            end
            S_CHK: begin
                mode <= 2'd0;
                if (bad) begin
                    {err_load}
                    err <= 1'b1; state <= S_SEND;
                end else begin
                    in_valid <= 1'b1; err <= 1'b0; state <= S_RUN;
                end
            end
            S_RUN: if (out_valid) begin
                {" ".join(load)}
                state <= S_SEND;
            end
            S_SEND: if (!tx_busy) begin tx_start <= 1'b1; state <= S_WAIT; end
            S_WAIT: state <= S_DONE;                   // let tx_busy rise
            S_DONE: if (!tx_busy) begin
                if (cnt == last) begin cnt <= 0; state <= S_CMD; end
                else begin cnt <= cnt + 1; state <= S_SEND; end
            end
            default: state <= S_CMD;
        endcase

        // drop a stalled request after TIMEOUT_CLKS of silence
        if (state == S_RECV && !rx_strobe) begin
            if (idle == TIMEOUT_CLKS - 1) begin
                idle <= 0; cnt <= 0; bad <= 1'b0; state <= S_CMD;
            end else idle <= idle + 1;
        end else idle <= 0;
    end

    assign leds = ~{{err, rep[0][4:0]}};                // LEDs are active low
endmodule
'''
    uart = Path(__file__).with_name("verilog_uart_modules.v").read_text()

    # ── testbench ──
    nv = len(bundle.vectors)
    stride = max(1, nv // 32)
    sampled = list(range(0, nv, stride))
    init = []
    for k, b in enumerate(info):
        init.append(f"        einfo[{k}] = 8'd{b};")
    for v, (il, ol) in enumerate(bundle.vectors):
        for k, x in enumerate(il):
            init.append(f"        vin[{v * ni + k}] = 8'd{x};")
        for k, x in enumerate(ol):
            init.append(f"        vout[{v * no + k}] = 8'd{x};")
    for si, v in enumerate(sampled):
        for k, x in enumerate(bundle.dbg_vectors[v]):
            init.append(f"        vdbg[{si * nd + k}] = 8'd{x};")
    ns = len(sampled)
    resync_partial = ""
    if ni >= 2:
        resync_partial = '''        // resync: command and first request byte, then silence
        send_byte("R"); send_byte(vin[0]);
        repeat (TO + 100) @(posedge clk);
        check_vec(0);
'''
    tb = f'''`timescale 1ns/1ps
// Whole-design test for {top} (debug protocol): bit-bangs the UART, reads the
// info frame, sends every vector with 'R' and compares the reply with the
// compiler's Python reference ({nv} vector{'s' * (nv != 1)}), reads the state
// back with 'S' after {ns} of them and compares all {nd} state bytes, then
// checks the out-of-range reply, an unknown command byte and the stall
// timeout. Uses 8 clocks per bit. Prints PASS or FAIL.
module {top}_tb;
    localparam CPB = 8, TO = 2000;
    localparam NI = {ni}, NO = {no}, ND = {nd}, NV = {nv}, NINFO = {INFO_LEN};
    localparam STRIDE = {stride};
    reg clk = 1'b0;
    reg rx  = 1'b1;
    wire tx;
    wire [5:0] leds;
    {top} #(.CLKS_PER_BIT(CPB), .TIMEOUT_CLKS(TO)) dut (.clk(clk), .rx(rx), .tx(tx), .leds(leds));
    always #5 clk = ~clk;

    reg [7:0] einfo [0:NINFO-1];
    reg [7:0] vin   [0:NV*NI-1];
    reg [7:0] vout  [0:NV*NO-1];
    reg [7:0] vdbg  [0:{ns}*ND-1];
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

    task check_info;
        integer j;
        reg [7:0] got;
        begin
            send_byte("I");
            for (j = 0; j < NINFO; j = j + 1) begin
                recv_byte(got);
                if (got !== einfo[j]) begin
                    $display("FAIL: info byte %0d = %0d, expected %0d", j, got, einfo[j]);
                    errors = errors + 1;
                end
            end
            checked = checked + 1;
        end
    endtask

    task check_vec;
        input integer idx;
        integer j;
        reg [7:0] got;
        begin
            send_byte("R");
            for (j = 0; j < NI; j = j + 1) send_byte(vin[idx * NI + j]);
            for (j = 0; j < NO; j = j + 1) begin
                recv_byte(got);
                if (got !== vout[idx * NO + j]) begin
                    $display("FAIL: vector %0d reply byte %0d = %0d, expected %0d",
                             idx, j, got, vout[idx * NO + j]);
                    errors = errors + 1;
                end
            end
            checked = checked + 1;
        end
    endtask

    task check_state;
        input integer idx;                          // index among sampled vectors
        integer j;
        reg [7:0] got;
        begin
            send_byte("S");
            for (j = 0; j < ND; j = j + 1) begin
                recv_byte(got);
                if (got !== vdbg[idx * ND + j]) begin
                    $display("FAIL: sample %0d state byte %0d = %0d, expected %0d",
                             idx, j, got, vdbg[idx * ND + j]);
                    errors = errors + 1;
                end
            end
            checked = checked + 1;
        end
    endtask

    task check_error;
        integer j;
        reg [7:0] got;
        begin
            send_byte("R");
            send_byte({p});                          // out of range
            for (j = 1; j < NI; j = j + 1) send_byte(0);
            for (j = 0; j < NO; j = j + 1) begin
                recv_byte(got);
                if (got !== 8'hFF) begin
                    $display("FAIL: error reply byte %0d = %0d, expected 255", j, got);
                    errors = errors + 1;
                end
            end
            checked = checked + 1;
        end
    endtask

    integer v;
    initial begin
{nl.join(init)}
        repeat (20) @(posedge clk);
        check_info;
        for (v = 0; v < NV; v = v + 1) begin
            check_vec(v);
            if (v % STRIDE == 0) check_state(v / STRIDE);
        end
        check_error;
        check_vec(0);                               // still works after an error
        send_byte(8'h7E);                           // unknown command: ignored
        check_info;
        // resync: a command and no request, then silence longer than the timeout
        send_byte("R");
        repeat (TO + 100) @(posedge clk);
        check_info;
{resync_partial}        check_state(0);                             // state is still readable
        if (errors == 0) $display("PASS");
        else $display("FAIL: %0d errors in %0d checks", errors, checked);
        $finish;
    end

    initial begin
        #200_000_000;
        $display("FAIL: timeout after %0d checks", checked);
        $finish;
    end
endmodule
'''
    return UartWrapper(source=uart + "\n" + top_src, testbench=tb,
                       top_name=top, request_bytes=ni, reply_bytes=no)

"""RTL generator for cmtz-core, plus a testbench checked against `isa/sim.py`.

One core per field: p, q and the memory sizes are fixed when the Verilog is
generated; programs are loaded at run time through the host ports. Plain
Verilog-2001 for iverilog and yosys, like the Verilog backend.

Design:
  - Multi-cycle FSM: FETCH (program memory has a synchronous read, so the
    instruction arrives one clock later), EXEC, and MEM for loads. A
    dispatched interrupt takes one extra FETCH cycle. CPI is not 1.
  - Field registers and index registers are flip-flop arrays (three read
    ports); program and data memory are written so yosys can map them to
    block RAM.
  - Reductions are unrolled conditional subtracts of p << k (or q << k), as
    in the Verilog backend; no `%`.
  - Decoding comes from `isa/encoding.py`, so the opcode numbers cannot
    drift from the assembler and the simulator.
  - Host ports (program memory, data memory, field registers, config) work
    only while the core is idle. `start` runs from the entry; `irq` sets the
    pending bit; `abort` stops a run (status error, code 7).

Run contract (same as `Machine.start`): `start` resets pc, return stack,
pending/in-handler/handled, status, error and the retired count; registers
and data memory keep their contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..isa.encoding import OPS, STACK_DEPTH
from ..isa.image import CoreConfig, ProgramImage
from ..isa.sim import ERROR, HALTED, Machine, RUNNING


def _bits(n: int) -> int:
    return max(1, n.bit_length())


def _reduce_fn(name: str, xw: int, mod: int) -> str:
    """Verilog function reducing any xw-bit value mod `mod`."""
    xmax = (1 << xw) - 1
    ow = _bits(mod - 1)
    lines = [f"function [{ow - 1}:0] {name};",
             f"    input [{xw - 1}:0] x;",
             f"    reg [{xw - 1}:0] t;",
             "    begin",
             "        t = x;"]
    if xmax >= mod:
        kmax = 0
        while (mod << (kmax + 1)) <= xmax:
            kmax += 1
        for k in range(kmax, -1, -1):
            m = mod << k
            lines.append(f"        if (t >= {xw}'d{m}) t = t - {xw}'d{m};")
    lines += [f"        {name} = t[{ow - 1}:0];", "    end", "endfunction"]
    return "\n".join(lines)


def _operand_use() -> str:
    """Verilog: which instruction fields (a, b, c) name field registers, per
    opcode, from `encoding.FORMATS` (what a lane checks against leases)."""
    from ..isa.encoding import FORMATS
    lines = ["always @* begin", "    case (op)"]
    for name, spec in OPS.items():
        flds = {f for k, f in FORMATS[spec.fmt] if k == "r"}
        bits = "".join("1" if f in flds else "0" for f in "abc")
        if bits != "000":
            lines.append(f"    OP_{name}: use_r = 3'b{bits};")
    lines += ["    default: use_r = 3'b000;", "    endcase", "end"]
    return "\n".join(lines)


def emit_core(core: CoreConfig, module_name: str = "cmtz_core",
              lane: bool = False) -> str:
    """The core. `lane=True` is the multilane variant (RFC-003): the field
    register file moves out of the module (read ports rf_d*, a registered
    write port rf_we/rf_wd for register rf_fa), the lane reports which
    fields name registers (`rf_use`) and whether it is in EXEC (`rf_ex`),
    and `stall` holds it in EXEC. The default output is unchanged."""
    p, q = core.p, core.q
    rw = core.rw
    pcw = _bits(core.pmem_size)             # pc may equal the program length
    paw = _bits(core.pmem_size - 1)
    daw = _bits(core.dmem_size - 1)
    spw = _bits(STACK_DEPTH)
    ops = {name: spec.code for name, spec in OPS.items()}
    lp = "\n".join(f"localparam OP_{n} = 6'd{c};" for n, c in ops.items())
    fns = "\n\n".join([
        _reduce_fn("red_add", rw + 1, p),
        _reduce_fn("red_mul", 2 * rw + 1, p),
        _reduce_fn("red_hq", 2 * rw + 1, q),
        _reduce_fn("red_redp", rw, p),
    ])
    if lane:
        rports = f"""    output wire [4:0]      rf_fa,
    output wire [4:0]      rf_fb,
    output wire [4:0]      rf_fc,
    output wire [2:0]      rf_use,
    output wire            rf_ex,
    input  wire [{rw - 1}:0]      rf_da,
    input  wire [{rw - 1}:0]      rf_db,
    input  wire [{rw - 1}:0]      rf_dc,
    output reg             rf_we,
    output reg  [{rw - 1}:0]      rf_wd,
    input  wire            stall,
"""
        rdecl = rinit = rassign = ""
        rsrc = ("rf_da", "rf_db", "rf_dc")
        wr_body = "rf_we <= (fa != 5'd0); rf_wd <= v;"
        host_w = "    rf_we <= 1'b0;\n"
        exec_head = "    S_EXEC: if (!stall) case (op)"
        lane_note = ("// Lane variant (RFC-003): the field registers live "
                     "outside, reached through\n// the rf_* ports; stall "
                     "holds the lane in EXEC.\n")
        use = ("\nreg [2:0] use_r;\n" + _operand_use() + "\n"
               "assign rf_fa = fa;\nassign rf_fb = fb;\nassign rf_fc = fc;\n"
               "assign rf_use = use_r;\nassign rf_ex = (state == S_EXEC);\n"
               f"initial begin rf_we = 1'b0; rf_wd = {rw}'d0; end\n")
    else:
        rports = f"""    input  wire            r_we,
    input  wire [4:0]      r_addr,
    input  wire [{rw - 1}:0]      r_wdata,
    output wire [{rw - 1}:0]      r_rdata,
"""
        rdecl = f"reg [{rw - 1}:0] R [0:31];\n"
        rinit = f"    for (i = 0; i < 32; i = i + 1) R[i] = {rw}'d0;\n"
        rassign = "assign r_rdata = R[r_addr];\n"
        rsrc = ("R[fa]", "R[fb]", "R[fc]")
        wr_body = "if (fa != 5'd0) R[fa] <= v;"
        host_w = "    if (!running && r_we && r_addr != 5'd0) R[r_addr] <= r_wdata;\n"
        exec_head = "    S_EXEC: case (op)"
        lane_note = use = ""
    return f"""// cmtz-core: reusable Cook-Mertz processor, F_{p} (q = {q}).
// Generated by cmtz/backends/isa_core.py from cmtz/isa/encoding.py
// (ISA version {core.isa_version}). Do not edit by hand.
//
// program memory {core.pmem_size} x 32, data memory {core.dmem_size} x {rw},
// R0..R31 x {rw} bits, X0..X15 x 16 bits, return stack {STACK_DEPTH}.
// status: 0 idle, 1 running, 2 halted, 3 error.
// error:  1 opcode, 2 pc, 3 stack overflow, 4 underflow, 5 RETI, 6 address,
//         7 aborted.
{lane_note}
module {module_name} (
    input  wire            clk,
    input  wire            start,
    input  wire            abort,
    input  wire            irq,
    input  wire [{pcw - 1}:0]     cfg_len,
    input  wire [{pcw - 1}:0]     cfg_entry,
    input  wire [{pcw - 1}:0]     cfg_handler,
    input  wire            cfg_has_handler,
    input  wire            pm_we,
    input  wire [{paw - 1}:0]     pm_addr,
    input  wire [31:0]     pm_wdata,
    input  wire            dm_we,
    input  wire [{daw - 1}:0]     dm_addr,
    input  wire [{rw - 1}:0]      dm_wdata,
    output wire [{rw - 1}:0]      dm_rdata,
{rports}    input  wire [3:0]      x_addr,
    output wire [15:0]     x_rdata,
    output wire [1:0]      status,
    output wire [2:0]      error,
    output reg  [31:0]     retired,
    output reg  [31:0]     disp_retired,
    output reg  [{pcw - 1}:0]     disp_pc,
    output reg  [15:0]     disp_count,
    output wire            boundary,
    output wire [{pcw - 1}:0]     pc_out
);

{lp}

localparam S_IDLE = 2'd0, S_FETCH = 2'd1, S_EXEC = 2'd2, S_MEM = 2'd3;
localparam ST_IDLE = 2'd0, ST_RUN = 2'd1, ST_HALT = 2'd2, ST_ERR = 2'd3;

{fns}

reg [1:0]  state = S_IDLE;
reg [1:0]  st = ST_IDLE;
reg [2:0]  err = 3'd0;
reg [{pcw - 1}:0] pc = {pcw}'d0;
reg [{pcw - 1}:0] saved_pc = {pcw}'d0;
reg        pending = 1'b0, in_h = 1'b0, handled = 1'b0;
reg [{spw - 1}:0] sp = {spw}'d0;
reg [{pcw - 1}:0] stack [0:{STACK_DEPTH - 1}];

{rdecl}reg [15:0] X [0:15];
reg [31:0] pm [0:{core.pmem_size - 1}];
reg [{rw - 1}:0] dm [0:{core.dmem_size - 1}];
reg [31:0] ir = 32'd0;
reg [{rw - 1}:0] dm_q = {rw}'d0;

integer i;
initial begin
{rinit}    for (i = 0; i < 16; i = i + 1) X[i] = 16'd0;
    for (i = 0; i < {STACK_DEPTH}; i = i + 1) stack[i] = {pcw}'d0;
    retired = 32'd0; disp_retired = 32'd0; disp_pc = {pcw}'d0;
    disp_count = 16'd0;
end

assign status = st;
assign error = err;
assign boundary = (state == S_FETCH);
assign pc_out = pc;
{rassign}assign x_rdata = X[x_addr];
assign dm_rdata = dm_q;

// ── decode ──────────────────────────────────────────────────────────
wire [5:0]  op  = ir[31:26];
wire [4:0]  fa  = ir[25:21];
wire [4:0]  fb  = ir[20:16];
wire [4:0]  fc  = ir[15:11];
wire [10:0] imm = ir[10:0];
wire [{rw - 1}:0] ra = {rsrc[0]};
wire [{rw - 1}:0] rb = {rsrc[1]};
wire [{rw - 1}:0] rc = {rsrc[2]};
wire [{rw - 1}:0] irw = imm[{rw - 1}:0];
wire [15:0] xa = X[fa[3:0]];
wire [15:0] xb = X[fb[3:0]];
wire [15:0] xc = X[fc[3:0]];
wire [15:0] simm = {{{{5{{imm[10]}}}}, imm}};
wire [15:0] maddr = xb + {{5'd0, imm}};
wire        addr_ok = (maddr < 16'd{core.dmem_size});
wire [{pcw - 1}:0] target = imm[{pcw - 1}:0];
wire [{pcw - 1}:0] pc1 = pc + {pcw}'d1;
{use}
// ── field ALU ───────────────────────────────────────────────────────
wire [{rw}:0]   add_s  = {{1'b0, rb}} + {{1'b0, rc}};
wire [{rw - 1}:0] add_r  = red_add(add_s);
wire [{rw}:0]   sub_s  = (rb >= rc) ? ({{1'b0, rb}} - {{1'b0, rc}})
                                 : ({{1'b0, rb}} + {rw + 1}'d{p} - {{1'b0, rc}});
wire [{rw - 1}:0] sub_r  = red_add(sub_s);
wire [{2 * rw}:0] mul_x  = rb * rc;
wire [{rw - 1}:0] mul_r  = red_mul(mul_x);
wire [{2 * rw}:0] mac_x  = ra + rb * rc;
wire [{rw - 1}:0] mac_r  = red_mul(mac_x);
wire [{2 * rw}:0] muli_x = rb * irw;
wire [{rw - 1}:0] muli_r = red_mul(muli_x);
wire [{2 * rw}:0] hq_x   = ra * rb + irw;
wire [{rw - 1}:0] hq_r   = red_hq(hq_x);
wire [{rw - 1}:0] redp_r = red_redp(rb);

// ── data memory: host port while idle, core port while running ─────
wire running = (state != S_IDLE);
wire [{daw - 1}:0] dm_a = running ? maddr[{daw - 1}:0] : dm_addr;
wire dm_w = running ? (state == S_EXEC && op == OP_ST && addr_ok) : dm_we;
wire [{rw - 1}:0] dm_d = running ? ra : dm_wdata;
always @(posedge clk) begin
    if (dm_w) dm[dm_a] <= dm_d;
    dm_q <= dm[dm_a];
end

// ── program memory: host writes while idle; the core reads at pc ────
always @(posedge clk) begin
    if (pm_we && !running) pm[pm_addr] <= pm_wdata;
    ir <= pm[pc[{paw - 1}:0]];
end

// ── control ─────────────────────────────────────────────────────────
task fail(input [2:0] code);
    begin
        st <= ST_ERR; err <= code; state <= S_IDLE;
    end
endtask

task wr(input [{rw - 1}:0] v);
    begin
        {wr_body}
    end
endtask

task wx(input [15:0] v);
    begin
        if (fa[3:0] != 4'd0) X[fa[3:0]] <= v;
    end
endtask

task advance;
    begin
        pc <= pc1; retired <= retired + 32'd1; state <= S_FETCH;
    end
endtask

task jump(input [{pcw - 1}:0] to);
    begin
        pc <= to; retired <= retired + 32'd1; state <= S_FETCH;
    end
endtask

always @(posedge clk) begin
    if (irq) pending <= 1'b1;
{host_w}    if (running && abort) begin
        fail(3'd7);
    end else case (state)
    S_IDLE: if (start) begin
        pc <= cfg_entry; sp <= {spw}'d0; pending <= 1'b0; in_h <= 1'b0;
        handled <= 1'b0; retired <= 32'd0; st <= ST_RUN; err <= 3'd0;
        state <= S_FETCH;
    end
    S_FETCH: begin
        if ((pending || irq) && !in_h) begin
            pending <= 1'b0;
            if (cfg_has_handler) begin
                saved_pc <= pc; pc <= cfg_handler; in_h <= 1'b1;
                disp_retired <= retired; disp_pc <= pc;
                disp_count <= disp_count + 16'd1;
            end
        end else if (pc >= cfg_len) begin
            fail(3'd2);
        end else begin
            state <= S_EXEC;
        end
    end
{exec_head}
        OP_NOP:    advance;
        OP_HALT:   begin retired <= retired + 32'd1; st <= ST_HALT; state <= S_IDLE; end
        OP_ADD:    begin wr(add_r); advance; end
        OP_SUB:    begin wr(sub_r); advance; end
        OP_MUL:    begin wr(mul_r); advance; end
        OP_MAC:    begin wr(mac_r); advance; end
        OP_MULI:   begin wr(muli_r); advance; end
        OP_MOV:    begin wr(rb); advance; end
        OP_LDI:    begin wr(irw); advance; end
        OP_HORNER: begin wr(hq_r); advance; end
        OP_REDP:   begin wr(redp_r); advance; end
        OP_LIX:    begin wx({{5'd0, imm}}); advance; end
        OP_ADDIX:  begin wx(xb + simm); advance; end
        OP_ADDX:   begin wx(xb + xc); advance; end
        OP_MULX:   begin wx(xb * xc); advance; end
        OP_FTOX:   begin wx({{{16 - rw}'d0, rb}}); advance; end
        OP_LD:     if (!addr_ok) fail(3'd6); else state <= S_MEM;
        OP_ST:     if (!addr_ok) fail(3'd6); else advance;
        OP_JMP:    jump(target);
        OP_BEQX:   jump(xa == xb ? target : pc1);
        OP_BNEX:   jump(xa != xb ? target : pc1);
        OP_BLTX:   jump(xa <  xb ? target : pc1);
        OP_BZ:     jump(ra == {rw}'d0 ? target : pc1);
        OP_BNZ:    jump(ra != {rw}'d0 ? target : pc1);
        OP_CALL:   if (sp == {spw}'d{STACK_DEPTH}) fail(3'd3);
                   else begin stack[sp[{_bits(STACK_DEPTH - 1) - 1}:0]] <= pc1; sp <= sp + {spw}'d1; jump(target); end
        OP_RET:    if (sp == {spw}'d0) fail(3'd4);
                   else begin sp <= sp - {spw}'d1; jump(stack[sp[{_bits(STACK_DEPTH - 1) - 1}:0] - 1'b1]); end
        OP_RETI:   if (!in_h) fail(3'd5);
                   else begin in_h <= 1'b0; handled <= 1'b1; jump(saved_pc); end
        OP_WAITI:  if (handled) advance; else state <= S_FETCH;
        default:   fail(3'd1);
    endcase
    S_MEM: begin wr(dm_q); advance; end
    endcase
end

endmodule
"""


# ── testbench ────────────────────────────────────────────────────────────────

@dataclass
class CoreCase:
    """One run on the core: load `image`, write inputs and matrices, run."""
    image: ProgramImage
    name: str = ""
    inputs: dict[str, int] = field(default_factory=dict)
    matrices: dict[str, list[list[int]]] = field(default_factory=dict)
    irq_at: tuple[int, ...] = ()
    max_cycles: int = 2_000_000
    # cycles after start before any trigger may fire; only meaningful for a
    # trigger at the count where main stalls on WAITI (the simulator has no
    # cycles, so elsewhere it would change what the hardware does)
    irq_delay: int = 0
    # also check the clock cycles from start to the end of the run against
    # the simulator's FSM model (`Machine.cycles`)
    check_cycles: bool = False


@dataclass
class CoreTestbench:
    source: str
    cases: int
    expected: list[Machine]         # simulator state after each case


def emit_core_testbench(core: CoreConfig, cases: list[CoreCase],
                        module_name: str = "cmtz_core",
                        break_alu: bool = False) -> CoreTestbench:
    """A self-checking testbench running `cases` back to back on one core.

    Expected values come from `isa/sim.py`, which carries R, X and data
    memory from case to case exactly as the core does. After each run the
    testbench checks status, error, retired count, every R and X register,
    the last dispatch, and every data word the simulator says changed.

    `break_alu` makes the expected ADD results wrong on purpose, to prove the
    testbench can print FAIL.
    """
    pcw = _bits(core.pmem_size)
    rw = core.rw
    lines: list[str] = []
    R, X, D = [0] * 32, [0] * 16, [0] * core.dmem_size
    expected = []
    for n, c in enumerate(cases):
        img = c.image
        img.validate(core)
        if len(c.irq_at) > 8:
            raise ValueError("at most 8 interrupt points per case")
        m = Machine(core, img)
        m.R, m.X, m.D = list(R), list(X), list(D)
        m.set_inputs(c.inputs)
        for k, v in c.matrices.items():
            m.write_matrix(k, v)
        before = list(m.D)
        writes_r = {img.inputs[k]: v for k, v in c.inputs.items()}
        writes_d = {a: v for a, (v, w) in enumerate(zip(m.D, D)) if v != w}
        m.run(irq_at=c.irq_at)
        if m.status not in (HALTED, ERROR):
            raise ValueError(f"case {n} ({c.name}) does not finish "
                             "(waiting on WAITI with no trigger)")
        expected.append(m)
        lines.append(f"        // case {n}: {c.name}")
        lines.append(f"        case_no = {n};")
        for a, w in enumerate(img.words):
            lines.append(f"        pmw({a}, 32'h{w:08x});")
        lines.append(f"        cfg_len = {len(img.words)}; cfg_entry = {img.entry}; "
                     f"cfg_handler = {img.handler or 0}; "
                     f"cfg_has_handler = {int(img.handler is not None)};")
        for r, v in writes_r.items():
            lines.append(f"        rw_({r}, {v});")
        for a, v in writes_d.items():
            lines.append(f"        dw({a}, {v});")
        lines.append(f"        irq_n = {len(c.irq_at)}; irq_delay = {c.irq_delay};")
        for j, k in enumerate(c.irq_at):
            lines.append(f"        irq_k[{j}] = {k};")
        lines.append(f"        run({c.max_cycles});")
        disp = m.dispatches[-1] if m.dispatches else None
        lines.append(f"        chk_status({m.status}, {m.error}, {m.retired});")
        if c.check_cycles:
            if not m.cycles_exact:
                raise ValueError(f"case {n} ({c.name}) stalls on WAITI; "
                                 "its cycle count depends on the trigger")
            lines.append(f"        chk_cycles({m.cycles});")
        if disp:
            lines.append(f"        chk_disp({disp[0]}, {disp[1]});")
        for r in range(32):
            v = m.R[r]
            if break_alu and any(i.startswith("ADD R") and f"R{r}," in i[:8]
                                 for i in img.listing):
                v = (v + 1) % (1 << rw)
            lines.append(f"        chk_r({r}, {v});")
        for x in range(16):
            lines.append(f"        chk_x({x}, {m.X[x]});")
        for a, (v, w) in enumerate(zip(m.D, before)):
            if v != w:
                lines.append(f"        chk_d({a}, {v});")
        R, X, D = m.R, m.X, m.D

    body = "\n".join(lines)
    chk_cycles = '''task chk_cycles(input integer n);
    if (cyc !== n) begin
        $display("FAIL: case %0d took %0d cycles, expected %0d", case_no, cyc, n);
        errors = errors + 1;
    end
endtask
''' if any(c.check_cycles for c in cases) else ""
    daw = _bits(core.dmem_size - 1)
    paw = _bits(core.pmem_size - 1)
    src = f"""// Testbench for {module_name}. Generated by cmtz/backends/isa_core.py;
// expected values from cmtz/isa/sim.py. Prints PASS or FAIL.
`timescale 1ns/1ps
module {module_name}_tb;
reg clk = 0;
always #5 clk = ~clk;
reg start = 0, abort = 0, irq = 0;
reg [{pcw - 1}:0] cfg_len = 0, cfg_entry = 0, cfg_handler = 0;
reg cfg_has_handler = 0;
reg pm_we = 0; reg [{paw - 1}:0] pm_addr = 0; reg [31:0] pm_wdata = 0;
reg dm_we = 0; reg [{daw - 1}:0] dm_addr = 0; reg [{rw - 1}:0] dm_wdata = 0;
wire [{rw - 1}:0] dm_rdata;
reg r_we = 0; reg [4:0] r_addr = 0; reg [{rw - 1}:0] r_wdata = 0;
wire [{rw - 1}:0] r_rdata;
wire [1:0] status; wire [2:0] error;
wire [31:0] retired, disp_retired; wire [{pcw - 1}:0] disp_pc, pc_out;
wire [15:0] disp_count; wire boundary;

{module_name} dut (
    .clk(clk), .start(start), .abort(abort), .irq(irq),
    .cfg_len(cfg_len), .cfg_entry(cfg_entry), .cfg_handler(cfg_handler),
    .cfg_has_handler(cfg_has_handler),
    .pm_we(pm_we), .pm_addr(pm_addr), .pm_wdata(pm_wdata),
    .dm_we(dm_we), .dm_addr(dm_addr), .dm_wdata(dm_wdata), .dm_rdata(dm_rdata),
    .r_we(r_we), .r_addr(r_addr), .r_wdata(r_wdata), .r_rdata(r_rdata),
    .x_addr(4'd0), .x_rdata(),
    .status(status), .error(error), .retired(retired),
    .disp_retired(disp_retired), .disp_pc(disp_pc), .disp_count(disp_count),
    .boundary(boundary), .pc_out(pc_out));

integer errors = 0, case_no = 0, irq_n = 0, irq_delay = 0, j, cyc;
integer irq_k [0:7];
reg fired [0:7];

task pmw(input integer a, input [31:0] w);
    begin pm_we = 1; pm_addr = a; pm_wdata = w; @(negedge clk); pm_we = 0; end
endtask
task rw_(input integer r, input integer v);
    begin r_we = 1; r_addr = r; r_wdata = v; @(negedge clk); r_we = 0; end
endtask
task dw(input integer a, input integer v);
    begin dm_we = 1; dm_addr = a; dm_wdata = v; @(negedge clk); dm_we = 0; end
endtask
task run(input integer maxc);
    begin
        for (j = 0; j < 8; j = j + 1) fired[j] = 0;
        start = 1; @(negedge clk); start = 0;
        cyc = 0;
        while (status == 2'd1 && cyc < maxc) begin
            irq = 0;
            for (j = 0; j < irq_n; j = j + 1)
                if (!fired[j] && boundary && retired == irq_k[j]
                        && cyc >= irq_delay) begin
                    irq = 1; fired[j] = 1;
                end
            @(negedge clk); cyc = cyc + 1;
        end
        irq = 0;
        if (status == 2'd1) begin
            $display("FAIL: case %0d still running after %0d cycles", case_no, cyc);
            errors = errors + 1;
            abort = 1; @(negedge clk); abort = 0;
        end
    end
endtask
task chk_status(input integer s, input integer e, input integer n);
    if (status !== s || error !== e || retired !== n) begin
        $display("FAIL: case %0d status %0d error %0d retired %0d, expected %0d %0d %0d",
                 case_no, status, error, retired, s, e, n);
        errors = errors + 1;
    end
endtask
task chk_disp(input integer n, input integer pc);
    if (disp_retired !== n || disp_pc !== pc) begin
        $display("FAIL: case %0d dispatch at retired %0d pc %0d, expected %0d %0d",
                 case_no, disp_retired, disp_pc, n, pc);
        errors = errors + 1;
    end
endtask
task chk_r(input integer r, input integer v);
    if (dut.R[r] !== v) begin
        $display("FAIL: case %0d R%0d = %0d, expected %0d", case_no, r, dut.R[r], v);
        errors = errors + 1;
    end
endtask
task chk_x(input integer x, input integer v);
    if (dut.X[x] !== v) begin
        $display("FAIL: case %0d X%0d = %0d, expected %0d", case_no, x, dut.X[x], v);
        errors = errors + 1;
    end
endtask
task chk_d(input integer a, input integer v);
    if (dut.dm[a] !== v) begin
        $display("FAIL: case %0d D[%0d] = %0d, expected %0d", case_no, a, dut.dm[a], v);
        errors = errors + 1;
    end
endtask
{chk_cycles}
initial begin
    repeat (2) @(negedge clk);
{body}
        if (errors == 0) $display("PASS: {len(cases)} cases");
        else $display("FAIL: %0d errors", errors);
        $finish;
end
endmodule
"""
    return CoreTestbench(src, len(cases), expected)


def standard_cases(core: CoreConfig, examples_dir, seed: int = 0) -> list[CoreCase]:
    """The regression set for an F_17 core: every real-field example
    compiled (random inputs, interrupts at random boundaries), matpow with
    2x2 and 3x3 matrices, the catalytic loop demo (both handlers), an
    exhaustive sweep of every field instruction, and each error code."""
    import random
    from pathlib import Path

    from ..compiler import compile_program
    from ..ir_nodes import IRMatPow
    from ..isa.asm import assemble
    from ..isa.demos import alu_sweep, catalytic_loop
    from .isa_backend import emit_isa

    rng = random.Random(seed)
    p = core.p
    cases: list[CoreCase] = []

    def rnd_matrix(n: int) -> list[list[int]]:
        return [[rng.randrange(p) for _ in range(n)] for _ in range(n)]

    for path in sorted(Path(examples_dir).glob("*.cmtz")):
        src = path.read_text()
        if "cfield" in src:
            continue
        r = compile_program(src, backend="python")
        if not r.ok or r.ir_program.ir_field.p != p:
            continue
        prog = r.ir_program
        mp = [n for _, n in prog.nodes() if isinstance(n, IRMatPow)]
        for n in ((2, 3) if mp else (2,)):
            b = emit_isa(prog, core, matrix_dims={m.matrix_name: n for m in mp})
            for _ in range(2):
                ins = {k: rng.randrange(p) for k in b.inputs}
                mats = {k: rnd_matrix(dim) for k, dim in b.matrices_in.items()}
                irq: tuple[int, ...] = ()
                if b.image.handler is not None:
                    probe = Machine(core, b.image)
                    probe.set_inputs(ins)
                    probe.run()
                    irq = (rng.randrange(probe.retired + 1),)
                    # and once with main already stalled on WAITI
                    cases.append(CoreCase(b.image, f"{path.stem} WAITI stall",
                                          ins, mats, (probe.retired,),
                                          irq_delay=200))
                cases.append(CoreCase(b.image, f"{path.stem} n={n}", ins,
                                      mats, irq))

    for restore in (True, False):
        img = catalytic_loop(restore=restore)
        for k in (0, 7, 33, 120):
            cases.append(CoreCase(img, f"catalytic_loop restore={restore} irq@{k}",
                                  {"x": rng.randrange(p)}, irq_at=(k,)))
    cases.append(CoreCase(catalytic_loop(), "catalytic_loop two irqs",
                          {"x": 5}, irq_at=(10, 12)))

    for op, k in [("ADD", 0), ("SUB", 0), ("MUL", 0), ("MAC", 0),
                  ("HORNER", 7), ("REDP", 0)] + [("MULI", k) for k in range(p)]:
        cases.append(CoreCase(alu_sweep(core, op, k), f"sweep {op} {k}"))

    def asm_case(name: str, src: str) -> CoreCase:
        a = assemble(src)
        return CoreCase(ProgramImage(core.p, core.q, a.words, a.entry,
                                     a.handler), name)
    cases += [
        asm_case("error stack overflow", "f: CALL f"),
        asm_case("error underflow", "RET"),
        asm_case("error reti", "RETI"),
        asm_case("error address", f"LIX X1, {core.dmem_size}\nLD R1, [X1]\nHALT"),
        asm_case("error pc", "NOP"),
        asm_case("R0 and X0 stay zero",
                 "LDI R0, 5\nADD R1, R0, R0\nLIX X0, 3\nADDIX X1, X0, 1\n"
                 "LIX X2, 4\nLD R0, [X2]\nMOV R3, R0\nHALT"),
        asm_case("call/ret and branches",
                 ".entry main\nf: LDI R1, 7\nRET\nmain: CALL f\nBZ R0, z\n"
                 "LDI R2, 1\nz: BNZ R1, y\nLDI R2, 2\ny: LIX X1, 2047\n"
                 "MULX X2, X1, X1\nADDIX X3, X0, -1\nFTOX X4, R1\nHALT"),
    ]
    return cases

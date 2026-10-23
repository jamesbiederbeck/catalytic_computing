"""Verilog generation for the ML-KEM NTT accelerator core.

``emit_core`` renders the datapath (banks, control ROM, crossbars, lanes, sequencer) from a
``NTTConfig`` and the two schedules.  Nothing here knows how to schedule; it only lays out
the words ``words.build_words`` produced.  The same generator emits every lane count.
"""
from .config import NTTConfig
from .sched import schedule
from .words import build_words


class Layout:
    """Bit layout of one control word."""

    def __init__(self, cfg):
        B, L, AW, BB = cfg.banks, cfg.lanes, cfg.addr_bits, cfg.bank_bits
        self.fields = []          # (name, width)
        for k in range(B):
            self.fields += [(f"raddr{k}", AW), (f"we{k}", 1), (f"waddr{k}", AW), (f"wsel{k}", BB)]
        for l in range(L):
            self.fields += [(f"lv{l}", 1), (f"asrc{l}", BB), (f"bsrc{l}", BB), (f"tw{l}", 12)]
        self.off, pos = {}, 0
        for name, w in self.fields:
            self.off[name] = (pos, w)
            pos += w
        self.width = pos

    def pack(self, word):
        v = 0
        for k in range(len(word.raddr)):
            for name, val in ((f"raddr{k}", word.raddr[k]), (f"we{k}", word.we[k]),
                              (f"waddr{k}", word.waddr[k]), (f"wsel{k}", word.wsel[k])):
                v |= val << self.off[name][0]
        for l in range(len(word.lane_v)):
            for name, val in ((f"lv{l}", word.lane_v[l]), (f"asrc{l}", word.asrc[l]),
                              (f"bsrc{l}", word.bsrc[l]), (f"tw{l}", word.tw[l])):
                v |= val << self.off[name][0]
        return v

    def wire(self, name):
        pos, w = self.off[name]
        return f"word[{pos + w - 1}:{pos}]" if w > 1 else f"word[{pos}]"


def build(cfg: NTTConfig):
    """(forward schedule, inverse schedule, forward words, inverse words)."""
    sf, si = schedule(cfg, False), schedule(cfg, True)
    return sf, si, build_words(sf), build_words(si)


def bank_expr(bm, idx, j):
    mask = 0
    for i in range(8):
        if (bm.cols[i] >> j) & 1:
            mask |= 1 << i
    return f"^({idx} & 8'b{mask:08b})"


def emit_core(cfg: NTTConfig):
    sf, si, wf, wi = build(cfg)
    bm = sf.bank_map
    lay = Layout(cfg)
    R = cfg.rom_reg
    flags = ["wv0"] + [f"wv{i}" for i in range(1, R)] + (["wv"] if R else [])
    wvname = flags[-1] if R else "wv0"
    flag_decl = "\n".join(f"    reg {f} = 1'b0;" for f in flags)
    if not R:
        flag_decl += "\n    wire wv = wv0;                       // the ROM word is executed as it comes out"
        wvname = "wv"
    chain = []
    prev = "word_r"
    for i in range(1, R):
        chain.append(f"        w{i} <= {prev};")
        prev = f"w{i}"
    if R:
        chain.append(f"        word_q <= {prev};")
    word_q_decl = "\n".join([f"    reg [WIDTH-1:0] w{i} = 0;" for i in range(1, R)]
                            + (["    reg [WIDTH-1:0] word_q = 0;"] if R else ["    wire [WIDTH-1:0] word_q = word_r;"]))
    B_ = cfg.banks
    src = flags[-2] if R else "wv0"
    if R:
        flag_decl += "\n" + "\n".join(f"    (* keep *) reg wvb{k} = 1'b0;      // per-bank copy of wv: keeps its fan-out local" for k in range(B_))
        chain += [f"        wvb{k} <= {src};" for k in range(B_)]
    else:
        flag_decl += "\n" + "\n".join(f"    wire wvb{k} = wv0;" for k in range(B_))
    wv_decl = flag_decl
    wv_assign = "\n".join(f"        {flags[i]} <= {flags[i - 1]};" for i in range(1, len(flags))) + ("\n" if len(flags) > 1 else "") + "\n".join(chain)
    wv_rst = "".join(f" {f} <= 1'b0;" for f in flags[1:]) + ("".join(f" wvb{k} <= 1'b0;" for k in range(cfg.banks)) if R else "")
    active_expr = " | ".join(["run"] + flags)
    others = " | ".join(["run"] + flags[:-1])
    last_cond = f"{wvname} && !({others})"

    B, L, AW, BB = cfg.banks, cfg.lanes, cfg.addr_bits, cfg.bank_bits
    nf, ni = len(wf), len(wi)
    total = nf + ni
    pcw = max(1, (total - 1).bit_length())
    rom = [lay.pack(w) for w in wf + wi]
    def bin_decl(k):
        if cfg.bin_reg:
            return (f"    reg [AW-1:0] ra_b{k} = 0, wa_b{k} = 0;\n    reg we_b{k} = 1'b0;\n    reg [11:0] wd_b{k} = 0;\n"
                    f"    always @(posedge clk) begin ra_b{k} <= ra{k}; wa_b{k} <= wa{k}; we_b{k} <= we{k}; wd_b{k} <= wd{k}; end")
        return (f"    wire [AW-1:0] ra_b{k} = ra{k}, wa_b{k} = wa{k};\n    wire we_b{k} = we{k};\n    wire [11:0] wd_b{k} = wd{k};")
    sdp = cfg.bank_prim == "sdp"
    n_rd = cfg.rd_reg - (1 if sdp else 0)       # fabric read stages (the SDP output register is one)

    def rd_chain_decl(k):
        return "\n".join(f"    reg [11:0] rd{k}_{i} = 0;      // read pipeline stage {i}" for i in range(1, n_rd + 1))

    def rd_out(k):
        return f"rd{k}_{n_rd}" if n_rd else f"rdm{k}"

    def mem_decl(k):
        if not sdp:
            return (f"    {ram_attr}reg [11:0] mem{k} [0:{cfg.depth - 1}];\n"
                    f"    reg [11:0] rdm{k} = 0;                // BSRAM read register")
        return f"    wire [35:0] sdo{k};\n    wire [11:0] rdm{k} = sdo{k}[11:0];"

    def mem_inst(k):
        if not sdp:
            return ""
        pad = 10 - cfg.addr_bits
        return (f"    // hand-instantiated simple-dual-port BSRAM, 18-bit words, READ_MODE=1: the block's own\n"
                f"    // output register carries one read stage\n"
                f"    SDPX9B #(.READ_MODE(1'b1), .BIT_WIDTH_0(18), .BIT_WIDTH_1(18), .BLK_SEL_0(3'b000),\n"
                f"             .BLK_SEL_1(3'b000), .RESET_MODE(\"SYNC\")) bank{k} (\n"
                f"        .CLKA(clk), .CEA(we_b{k}), .RESETA(1'b0),\n"
                f"        .ADA({{{pad}'d0, wa_b{k}, 4'b0011}}), .DI({{24'd0, wd_b{k}}}),\n"
                f"        .CLKB(clk), .CEB(1'b1), .RESETB(1'b0), .OCE(1'b1),\n"
                f"        .ADB({{{pad}'d0, ra_b{k}, 4'b0000}}), .DO(sdo{k}),\n"
                f"        .BLKSELA(3'b000), .BLKSELB(3'b000));")

    def mem_body(k):
        if sdp:
            return ""
        return f"        if (we_b{k}) mem{k}[wa_b{k}] <= wd_b{k};\n        rdm{k} <= mem{k}[ra_b{k}];"

    def rd_chain_body(k):
        prev, out = f"rdm{k}", ""
        for i in range(1, n_rd + 1):
            out += f"\n        rd{k}_{i} <= {prev};"
            prev = f"rd{k}_{i}"
        return out
    rom_attr = '(* ram_style = "logic" *) ' if cfg.rom_ram == 'logic' else ''
    ram_attr = '(* ram_style = "block" *) ' if cfg.bank_ram == 'block' else ''
    out = []
    p = out.append
    p(f"""\
// GENERATED by cmtz.accel.mlkem_ntt.rtlgen -- do not edit; regenerate from the Python source.
// ML-KEM (FIPS 203) 256-point NTT / inverse NTT accelerator core.
//   lanes = {L}, reduction = {cfg.reduction}, banks = {B} x {cfg.depth} x 12, in place
//   forward words = {nf}, inverse words = {ni}; kernel cycles = words + {cfg.overhead}
//   timing (clocks after a read issues): control +{cfg.t_ctl}, lane inputs +{cfg.t_in}, lane outputs +{cfg.t_out},
//   bank write +{cfg.t_wr}, dependent read allowed at +{cfg.dist}.
// Host port: when idle, h_idx/h_we/h_wdata access the coefficients (natural index order);
// h_rdata is valid one clock after h_idx is presented.
module mlkem_ntt_core (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,         // one-clock pulse, ignored while busy
    input  wire        inv,           // 0 forward, 1 inverse; sampled with start
    output wire        busy,
    output reg         done = 1'b0,   // one-clock pulse after the last write has landed
    output wire [15:0] cycles,        // clocks from the start edge to the done edge
    input  wire        h_we,
    input  wire [7:0]  h_idx,
    input  wire [11:0] h_wdata,
    output wire [11:0] h_rdata
);
    localparam B = {B}, L = {L}, AW = {AW}, BB = {BB};
    localparam NW_F = {nf}, NW_I = {ni}, TOTAL = {total};
    localparam WIDTH = {lay.width}, PCW = {pcw};

    // ---------------------------------------------------------------- control ROM
    {rom_attr}reg [WIDTH-1:0] rom [0:TOTAL-1];
    initial begin""")
    for i, v in enumerate(rom):
        p(f"        rom[{i}] = {lay.width}'h{v:x};")
    p(f"""\
    end
{wv_decl}
    reg [WIDTH-1:0] word_r = 0;                    // ROM output (BSRAM)
{word_q_decl}
    wire [WIDTH-1:0] word = word_q;     // idle reads ROM word 0, which issues nothing

    // ---------------------------------------------------------------- sequencer
    // start in cycle c0 -> run (ROM addresses) in c0+1 .. c0+NW, word k executes in
    // c0+{2 + cfg.rom_reg}+k, the last word in c0+NW+{1 + cfg.rom_reg} and done is high in
    // c0+NW+{2 + cfg.rom_reg}: cycles = NW + {2 + cfg.rom_reg}.
    reg run = 1'b0, inv_r = 1'b0;
    reg [{pcw - 1}:0] pc = 0, end_pc = 0;
    reg [15:0] cyc = 0;
    wire active = {active_expr};
    assign busy = active | done;
    assign cycles = cyc;
    always @(posedge clk) begin
        done <= 1'b0;
        // address gated by `run`: a real gate between the pc flip-flops and the block-RAM
        // address pins (a direct flip-flop drive fails nextpnr's Gowin BSRAM hold check)
        word_r <= rom[pc & {{PCW{{run}}}}];
        wv0 <= run;
{wv_assign}
        if (rst) begin
            run <= 1'b0; wv0 <= 1'b0;{wv_rst}
        end else if (start && !active) begin          // also accepted in the done clock
            run <= 1'b1; inv_r <= inv; cyc <= 16'd1;
            pc     <= inv ? NW_F : 0;
            end_pc <= inv ? TOTAL - 1 : NW_F - 1;
        end else if (active) begin
            cyc <= cyc + 1'b1;
            if (run) begin
                pc <= pc + 1'b1;
                if (pc == end_pc) run <= 1'b0;
            end
            if ({last_cond}) done <= 1'b1;
        end
    end

    // ---------------------------------------------------------------- host address decode
    wire [BB-1:0] h_bank = {{""")
    p(", ".join(f"{bank_expr(bm, 'h_idx', j)}" for j in reversed(range(BB))) + "};")
    addr_bits = ", ".join(f"h_idx[{i}]" for i in reversed(bm.addr_pos))
    p(f"    wire [AW-1:0] h_addr = {{{addr_bits}}};")
    hq = ["h_bank"] + [f"h_bank_q{i}" for i in range(1, cfg.rd_lat + 1)]
    p("    reg  [BB-1:0] " + ", ".join(f"{n} = 0" for n in hq[1:]) + ";\n    always @(posedge clk) begin " + " ".join(f"{hq[i + 1]} <= {hq[i]};" for i in range(cfg.rd_lat)) + " end")
    p(f"""
    // ---------------------------------------------------------------- banks
    wire [12*B-1:0] rd_flat;
    wire [12*{2 * L}-1:0] outs_flat;""")
    for k in range(B):
        p(f"""\
{mem_decl(k)}
{rd_chain_decl(k)}
    reg [11:0] wdr{k} = 0;                 // write-crossbar register
    wire [AW-1:0] ra{k} = wvb{k} ? {lay.wire(f'raddr{k}')} : h_addr;
    wire [BB-1:0] wsel{k} = {lay.wire(f'wsel{k}')};
    wire          ctl_we{k} = {lay.wire(f'we{k}')};
    wire [11:0]   wdsel{k} = outs_flat[wsel{k} * 12 +: 12];
    wire [11:0]   wd{k}  = wvb{k} ? {"wdr" + str(k) if cfg.wreg else "wdsel" + str(k)} : h_wdata;
    wire          we{k}  = wvb{k} ? ctl_we{k} : (h_we & (h_bank == {k}));
    wire [AW-1:0] wa{k}  = wvb{k} ? {lay.wire(f'waddr{k}')} : h_addr;
{bin_decl(k)}
{mem_inst(k)}
    always @(posedge clk) begin
{mem_body(k)}{rd_chain_body(k)}
        wdr{k} <= wdsel{k};
    end
    assign rd_flat[{12 * k + 11}:{12 * k}] = {rd_out(k)};""")
    p(f"    assign h_rdata = rd_flat[h_bank_q{cfg.rd_lat} * 12 +: 12];")
    p("\n    // ---------------------------------------------------------------- lanes")
    for l in range(L):
        a = f"rd_flat[{lay.wire(f'asrc{l}')} * 12 +: 12]"
        b = f"rd_flat[{lay.wire(f'bsrc{l}')} * 12 +: 12]"
        if cfg.xreg:
            p(f"""\
    reg [11:0] a_in{l} = 0, b_in{l} = 0, z_in{l} = 0;
    reg        v_in{l} = 1'b0;
    always @(posedge clk) begin
        a_in{l} <= {a};
        b_in{l} <= {b};
        z_in{l} <= {lay.wire(f'tw{l}')};
        v_in{l} <= {lay.wire(f'lv{l}')} & wv;
    end""")
        else:
            p(f"""\
    wire [11:0] a_in{l} = {a};
    wire [11:0] b_in{l} = {b};
    wire [11:0] z_in{l} = {lay.wire(f'tw{l}')};
    wire        v_in{l} = {lay.wire(f'lv{l}')} & wv;""")
        p(f"""\
    wire        vo{l};
    wire [11:0] o0_{l}, o1_{l};
    mlkem_bfly lane{l} (.clk(clk), .rst(rst), .v_in(v_in{l}), .mode(inv_r), .a(a_in{l}), .b(b_in{l}),
                        .z(z_in{l}), .v_out(vo{l}), .o0(o0_{l}), .o1(o1_{l}));
    assign outs_flat[{24 * l + 11}:{24 * l}] = o0_{l};
    assign outs_flat[{24 * l + 23}:{24 * l + 12}] = o1_{l};""")
    p("endmodule")
    return "\n".join(out) + "\n", dict(forward_words=nf, inverse_words=ni, width=lay.width,
                                        total=total)

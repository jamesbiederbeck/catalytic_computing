"""Draw a project's datapath as a plain SVG, from the compiler's hardware manifest.

    UART rx -> request bytes -> the program's DAG (one pipeline stage per
    column) -> reply bytes -> UART tx

Same visual language as the tang-nano-20k docs diagrams: plain shapes and text
with inline attributes only (no <style>, classes, scripts, fonts or images), a
white background rect of its own because GitHub draws an SVG as an image that
ignores dark mode, a monospace stack, <title> and aria-label, and a white rect
behind every label instead of a text-stroke halo.

Layout: each drawn node gets its own row and its column is its pipeline stage,
so an edge (out of the parent's row, then down a lane just left of the child's
column) never passes behind another box.
"""

from __future__ import annotations

import math

INK, TEAL, SOFT, MUTED, RULE, BG = "#1b2a2d", "#0b7a75", "#d9efec", "#5b6d70", "#c9d4d2", "#ffffff"
FONT = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'

CW = 7.3                      # width of one 12 px monospace character
ROW_H, BOX_H = 46, 38
GAP = 28                      # between columns: room for an edge lane
PAD = 20
HEAD = 78                     # title and column labels

OP_SHORT = {
    "IREmbed": "embed", "IRComplexEmbed": "cembed", "IRRotate": "rotate",
    "IRAdd": "add", "IRMul": "mul", "IRConjugate": "conj",
    "IRMagnitudeSq": "magsq", "IRPoly": "poly",
}


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


class Canvas:
    def __init__(self):
        self.p: list[str] = []

    def text(self, x, y, s, size=12, weight="normal", fill=INK, anchor="start"):
        self.p.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{FONT}\' font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')

    def label(self, x, y, s, size=11, fill=MUTED):
        """Small centred text on a white rect, so a wire never runs through it."""
        w = len(s) * size * 0.6 + 8
        self.p.append(f'<rect x="{x - w / 2:.1f}" y="{y - size:.1f}" width="{w:.1f}" '
                      f'height="{size + 5}" fill="{BG}" stroke="none"/>')
        self.text(x, y, s, size, "normal", fill, "middle")

    def box(self, x, y, w, h, kind="blk"):
        fill, stroke, dash = {"blk": (BG, INK, ""), "io": (SOFT, TEAL, ""),
                              "const": (BG, MUTED, "5 3")}[kind]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{d}/>')

    def frame(self, x, y, w, h):
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
                      f'fill="none" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5 4"/>')

    def arrow(self, pts, kind="ink", head=8):
        col = TEAL if kind == "acc" else INK
        wd = 2.2 if kind == "acc" else 1.4
        pl = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.p.append(f'<polyline points="{pl}" fill="none" stroke="{col}" stroke-width="{wd}" '
                      f'stroke-linejoin="round"/>')
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        a = math.atan2(y1 - y0, x1 - x0)
        bx, by = x1 - head * math.cos(a), y1 - head * math.sin(a)
        nx, ny = -math.sin(a) * head * 0.45, math.cos(a) * head * 0.45
        self.p.append(f'<polygon points="{x1:.1f},{y1:.1f} {bx + nx:.1f},{by + ny:.1f} '
                      f'{bx - nx:.1f},{by - ny:.1f}" fill="{col}"/>')

    def io_box(self, x, y, title, lines):
        w = max([len(title) * 8.6] + [len(s) * CW for s in lines]) + 28
        h = 22 + 18 * (1 + len(lines))
        self.box(x, y, w, h, "io")
        self.text(x + 14, y + 23, title, 14, "bold", TEAL)
        for i, s in enumerate(lines):
            self.text(x + 14, y + 23 + 19 * (i + 1), s, 12, "normal", MUTED)
        return w, h


def _value(v) -> str:
    return f"({v[0]},{v[1]})" if isinstance(v, (list, tuple)) else str(v)


def _lane_lines(ports, node_by_name, prefix: str) -> list[str]:
    """'byte k  name[.re|.im]' for each lane of the given ports."""
    out, k = [], 0
    for pt in ports:
        name = pt.node
        for lane in (("re", "im") if pt.is_complex else ("",)):
            out.append(f"{prefix} {k}  {name}{'.' + lane if lane else ''}")
            k += 1
    return out


def draw_datapath(bundle, title: str, subtitle: str = "") -> str:
    """SVG text for `bundle` (a VerilogBundle, debug or not)."""
    nodes = {n.name: n for n in bundle.nodes}
    order = [n.name for n in bundle.nodes]

    def source(name: str) -> str:
        """The node that actually holds an alias's value."""
        n = nodes[name]
        while n.kind == "alias" and n.parents:
            n = nodes[n.parents[-1] if n.op == "IRCompose" else n.parents[0]]
        return n.name

    outs = [pt for pt in bundle.ports if pt.direction == "output"]
    ins = [pt for pt in bundle.ports if pt.direction == "input"]

    # Synthesis drops nodes no output depends on; a debug build keeps them
    # alive through the state bus, so only then are they drawn.
    live: set[str] = set()
    stack = [source(pt.node) for pt in outs]
    while stack:
        n = stack.pop()
        if n not in live:
            live.add(n)
            stack += nodes[n].parents
    keep = (lambda n: True) if bundle.debug else (lambda n: n in live)

    runtime = {n for n in order if nodes[n].kind in ("input", "reg") and keep(n)}
    # constants only matter when something runtime reads them
    consts = {n for n in order if nodes[n].kind == "const"
              and any(n in nodes[c].parents for c in runtime)}
    drawn = [n for n in order if n in runtime or n in consts]
    folded = sum(1 for n in order if nodes[n].kind == "const" and n not in consts)


    # byte numbers on the wire, in port order (complex = re then im)
    def byte_map(ports):
        m, k = {}, 0
        for pt in ports:
            n = 2 if pt.is_complex else 1
            m[pt.node] = list(range(k, k + n))
            k += n
        return m
    in_bytes, out_bytes = byte_map(ins), byte_map(outs)

    def bytes_text(prefix, idx):
        return (f"{prefix} byte {idx[0]}" if len(idx) == 1
                else f"{prefix} bytes {idx[0]},{idx[1]}")

    # ── geometry ──
    def label_of(n):
        info = nodes[n]
        top = n
        sub = (f"const {_value(info.const)}" if info.kind == "const"
               else bytes_text("req", in_bytes[n]) if info.kind == "input" and n in in_bytes
               else OP_SHORT.get(info.op, info.op.replace("IR", "").lower()))
        return top, sub

    ncols = max((nodes[n].depth for n in drawn), default=0) + 1
    colw = [0.0] * ncols
    for n in drawn:
        top, sub = label_of(n)
        colw[nodes[n].depth] = max(colw[nodes[n].depth], _bw(None, (top, sub)))
    pill_sub = {pt.node: bytes_text("reply", out_bytes[pt.node]) for pt in outs}
    pill_w = max([_bw(None, (pt.node, pill_sub[pt.node])) for pt in outs] + [60])

    rx_lines = ["pin 70", "115200 8N1"] + (["cmds I R S"] if bundle.debug else [])
    tx_lines = ["pin 69", "115200 8N1"]
    tmp = Canvas()
    rx_w, rx_h = tmp.io_box(0, 0, "UART rx", rx_lines)
    tx_w, tx_h = tmp.io_box(0, 0, "UART tx", tx_lines)
    LINK = 64

    fx = PAD + rx_w + LINK                     # frame left
    x0 = fx + 24
    colx, x = [], x0
    for w in colw:
        colx.append(x)
        x += w + GAP
    pill_x = x                                  # first free x after the last column
    fw = (pill_x + pill_w + 20) - fx
    row_of = {n: i for i, n in enumerate(drawn)}
    pills: list[tuple[str, int]] = []           # (output port name, row)
    nrows = len(drawn)
    taken = set()
    for pt in outs:
        s = source(pt.node)
        r = row_of.get(s)
        if r is None or r in taken:             # constant output, or second output of a node
            r = nrows
            nrows += 1
        taken.add(r)
        pills.append((pt.node, r))
    top_y = HEAD + 18
    frame_h = nrows * ROW_H + 24
    width = fx + fw + LINK + tx_w + PAD
    height = top_y + frame_h + 46 + (18 if folded else 0)

    cv = Canvas()
    cv.text(PAD, 28, title, 16, "bold", INK)
    if subtitle:
        cv.text(PAD, 48, subtitle, 12, "normal", MUTED)
    cv.frame(fx, top_y - 6, fw, frame_h)

    def node_xy(n):
        i = nodes[n].depth
        return colx[i], top_y + 6 + row_of[n] * ROW_H + (ROW_H - BOX_H) / 2

    # column labels
    seen = set()
    for n in drawn:
        i = nodes[n].depth
        if i in seen:
            continue
        seen.add(i)
        lab = "ports" if i == 0 else f"stage {i}"
        cv.text(colx[i] + colw[i] / 2, top_y - 12, lab, 11, "normal", MUTED, "middle")

    # edges first, so boxes sit on top
    for n in drawn:
        if n in consts:
            continue
        cx, cy = node_xy(n)
        cy += BOX_H / 2
        for par in nodes[n].parents:
            if par not in row_of:
                continue
            px, py = node_xy(par)
            pr, py = px + _bw(nodes[par], label_of(par)), py + BOX_H / 2
            lane = cx - 12
            cv.arrow([(pr, py), (lane, py), (lane, cy), (cx, cy)])
            k = nodes[n].depth - 1 - nodes[par].depth
            if k >= 1 and nodes[par].kind != "const":
                cv.label((pr + lane) / 2, py - 5, f"delay {k}", 10)
    # outputs
    for name, r in pills:
        s = source(name)
        py = top_y + 6 + r * ROW_H + ROW_H / 2
        if s in row_of:
            sx, sy = node_xy(s)
            sr = sx + _bw(nodes[s], label_of(s))
            sy += BOX_H / 2
            lane = pill_x - 12
            cv.arrow([(sr, sy), (lane, sy), (lane, py), (pill_x, py)], "acc")
    # boxes
    for n in drawn:
        x, y = node_xy(n)
        info = nodes[n]
        top, sub = label_of(n)
        w = _bw(info, (top, sub))
        kind = "io" if info.kind == "input" else ("const" if info.kind == "const" else "blk")
        cv.box(x, y, w, BOX_H, kind)
        cv.text(x + 9, y + 16, top, 12, "bold", TEAL if kind == "io" else INK)
        cv.text(x + 9, y + 31, sub, 11, "normal", MUTED)
    for name, r in pills:
        py = top_y + 6 + r * ROW_H + ROW_H / 2
        cv.box(pill_x, py - BOX_H / 2, pill_w, BOX_H, "io")
        cv.text(pill_x + 9, py - BOX_H / 2 + 16, name, 12, "bold", TEAL)
        cv.text(pill_x + 9, py - BOX_H / 2 + 31, pill_sub[name], 11, "normal", MUTED)

    # UART boxes and the two links
    mid = top_y - 6 + frame_h / 2
    rx = Canvas()
    rx.io_box(PAD, mid - rx_h / 2, "UART rx", rx_lines)
    tx = Canvas()
    tx.io_box(fx + fw + LINK, mid - tx_h / 2, "UART tx", tx_lines)
    cv.p += rx.p + tx.p
    cv.arrow([(PAD + rx_w, mid), (fx, mid)], "acc")
    cv.label(PAD + rx_w + LINK / 2, mid - 8, "request", 11, TEAL)
    cv.arrow([(fx + fw, mid), (fx + fw + LINK, mid)], "acc")
    cv.label(fx + fw + LINK / 2, mid - 8, "reply", 11, TEAL)

    foot = f"{bundle.module_name}.v: {bundle.latency} pipeline stage{'s' if bundle.latency != 1 else ''}, one result per clock"
    if folded:
        foot += f"; {folded} constant node{'s' if folded != 1 else ''} folded at compile time"
    cv.text(fx, top_y - 6 + frame_h + 22, foot, 11, "normal", MUTED)

    desc = (f"{title}: UART receive pin 70 feeds {len(ins)} input port"
            f"{'s' if len(ins) != 1 else ''} ({', '.join(pt.node for pt in ins)}); the datapath has "
            f"{len([n for n in drawn if n in runtime and nodes[n].kind == 'reg'])} registered node"
            f"s over {bundle.latency} stage{'s' if bundle.latency != 1 else ''}; outputs "
            f"{', '.join(pt.node for pt in outs)} go to UART transmit pin 69.")
    # text can run past the drawing (footer, subtitle): never clip it
    width = max(width, fx + len(foot) * 6.7 + PAD, PAD + len(subtitle) * 7.3 + PAD,
                PAD + len(title) * 9.6 + PAD)
    w, h = math.ceil(width), math.ceil(height)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{esc(desc)}">\n<title>{esc(title)}</title>\n'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{BG}"/>\n'
            + "\n".join(cv.p) + "\n</svg>\n")


def _bw(info, labels) -> float:
    top, sub = labels
    return max(len(top) * 7.4, len(sub) * 6.7) + 18

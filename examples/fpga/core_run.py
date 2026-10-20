"""Check a cmtz-core bitstream on the board against the simulator and the
reference backend.

Usage: uv run python examples/fpga/core_run.py [--port /dev/serial/by-id/...-if01-port0]
                                              [--quick] [--only SECTION ...]

Before each run it reads the board's whole state (registers and data
memory), seeds the simulator with it, runs the same program on both and
compares the whole state afterwards, plus the program outputs against
`evaluate_all` / `eval_matpow_matrix`. Covers every real-field F_17 example,
matpow, an exhaustive sweep of every field instruction, an interrupt
sent over the UART while a long loop runs (the handler borrows a register
the loop is reading; the loop's result must not change), and the
roots-of-unity projection programs checked against
`cmtz/projection_oracle.py` as well (`projection()`). `soundness()`
reproduces, on the chip, what register values outside F_p do.

Reads and writes the serial port only; load the bitstream first
(openFPGALoader -c ft2232 _build/default/hardware.fs). Exit status 0 only if
everything matched.
"""

import argparse
import glob
import itertools
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # uart_check
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # cmtz

from uart_check import Port

from cmtz.compiler import compile_program
from cmtz.backends.isa_backend import emit_isa
from cmtz.backends.python_ref import PythonBackend
from cmtz.backends.verilog_backend import evaluate_all
from cmtz.ir_nodes import IRMatPow
from cmtz.isa import CoreConfig, Machine
from cmtz.isa import sim as S
from cmtz.isa import projection as PR
from cmtz.isa.demos import (
    alu_sweep, catalytic_loop, noncanonical_handler, sweep_expected,
)
from cmtz.isa.host import CoreLink
from cmtz.projection_oracle import gate, gate_tree

EXAMPLES = Path(__file__).resolve().parent.parent
CORE = CoreConfig(17, 19)


class Serial:
    """Port with the read(n) / write(data) interface CoreLink wants."""

    def __init__(self, port: Port):
        self.port = port

    def read(self, n: int) -> bytes:
        return self.port.read(n, timeout=0.5)

    def write(self, data: bytes) -> None:
        self.port.write(data)


class Board:
    def __init__(self, link: CoreLink):
        self.link = link
        self.fails = 0
        self.checks = 0

    def state(self):
        R, X = self.link.regs()
        D = self.link.read_data(0, CORE.dmem_size)
        return R, X, D

    def check(self, what: str, ok: bool, detail: str = "") -> bool:
        self.checks += 1
        if not ok:
            self.fails += 1
            print(f"  MISMATCH {what} {detail}")
        return ok

    def run(self, image, name, inputs=None, matrices=None,
            trigger_after=None, irq_at=()):
        """Run on the board and in the simulator from the board's state;
        returns (board status, simulator machine)."""
        R, X, D = self.state()
        m = Machine(CORE, image)
        m.R, m.X, m.D = R, X, D
        m.set_inputs(inputs or {})
        for k, v in (matrices or {}).items():
            m.write_matrix(k, v)
        st = self.link.run(image, inputs, matrices, trigger_after=trigger_after)
        m.run(irq_at=irq_at, max_steps=10_000_000)
        R2, X2, D2 = self.state()
        self.check(f"{name}: status", (st.status, st.error) == (m.status, m.error),
                   f"board {st.status}/{st.error}, sim {m.status}/{m.error}")
        self.check(f"{name}: retired", st.retired == m.retired,
                   f"board {st.retired}, sim {m.retired}")
        self.check(f"{name}: registers", (R2, X2) == (m.R, m.X))
        diff = [a for a in range(CORE.dmem_size) if D2[a] != m.D[a]]
        self.check(f"{name}: data memory", not diff, f"{len(diff)} words differ")
        return st, m


def examples(b: Board, rng: random.Random, vectors: int) -> None:
    for path in sorted(EXAMPLES.glob("*.cmtz")):
        src = path.read_text()
        if "cfield" in src:
            continue
        r = compile_program(src, backend="python")
        prog = r.ir_program
        if not r.ok or prog.ir_field.p != CORE.p:
            continue
        mp = [n for _, n in prog.nodes() if isinstance(n, IRMatPow)]
        for n in ((2, 3) if mp else (2,)):
            bundle = emit_isa(prog, CORE,
                              matrix_dims={m.matrix_name: n for m in mp})
            img = bundle.image
            t0 = time.monotonic()
            b.link.load(img)
            load_ms = 1000 * (time.monotonic() - t0)
            fails = b.fails
            for _ in range(vectors):
                ins = {k: rng.randrange(CORE.p) for k in bundle.inputs}
                mats = {k: [[rng.randrange(CORE.p) for _ in range(d)]
                            for _ in range(d)]
                        for k, d in bundle.matrices_in.items()}
                if img.handler is not None:
                    # main stalls on WAITI long before T arrives: the
                    # simulator's trigger lands at the stall
                    probe = Machine(CORE, img)
                    probe.scramble(0)
                    probe.set_inputs(ins)
                    probe.run()
                    st, m = b.run(img, path.stem, ins, mats, trigger_after=0.0,
                                  irq_at=(probe.retired,))
                else:
                    st, m = b.run(img, path.stem, ins, mats)
                exp = evaluate_all(prog, prog.ir_field, ins)
                got = b.link.outputs(img)
                b.check(f"{path.stem}: outputs",
                        got == {o: exp[o] % CORE.p for o in bundle.outputs},
                        f"{got}")
                for node in mp:
                    want = PythonBackend(prog.ir_field).eval_matpow_matrix(
                        node, mats[node.matrix_name])
                    b.check(f"{path.stem}: matpow {n}x{n}",
                            b.link.read_matrix(img, node.name) == want)
            extra = ""
            for node in mp:
                node.n_dim = n
                bound = node.cost_bound()["basic_instructions"]
                extra = (f", matpow {n}x{n} retired {st.retired} "
                         f"(Theorem 1.2 bound {bound:.0f})")
            print(f"{path.stem:28} {len(img.words):3} words, load "
                  f"{load_ms:5.1f} ms, {vectors} runs"
                  f"{' OK' if b.fails == fails else ' FAILED'}{extra}")


def sweeps(b: Board) -> None:
    ops = [("ADD", 0), ("SUB", 0), ("MUL", 0), ("MAC", 0), ("REDP", 0)] \
        + [("HORNER", k) for k in range(CORE.q)] \
        + [("MULI", k) for k in range(CORE.p)]
    fails = b.fails
    for op, k in ops:
        img = alu_sweep(CORE, op, k)
        b.run(img, f"sweep {op} {k}")
        want = sweep_expected(CORE, op, k)
        got = b.link.read_data(0, len(want))
        b.check(f"sweep {op} {k}", got == want,
                f"{sum(g != w for g, w in zip(got, want))} of {len(want)} wrong")
    print(f"{'instruction sweeps':28} {len(ops)} programs, every operand pair"
          f"{' OK' if b.fails == fails else ' FAILED'}")


def interrupts(b: Board, quick: bool) -> None:
    # Main must still be running when T arrives: a host round trip through
    # the debugger is about 6 ms, and T goes out only after G's reply. At
    # about 2 clocks per instruction, 1000 x 300 runs ~90 ms, 2000 x 300 ~180.
    outer, inner = (1000, 300) if quick else (2000, 300)
    good = catalytic_loop(outer, inner)
    bad = catalytic_loop(outer, inner, restore=False)
    fails = b.fails
    late = 0
    for x in (3, 11):
        st0, m0 = b.run(good, "catalytic loop, no T", {"x": x})
        acc0 = m0.outputs()["acc"]
        landed = []
        for delay in (0.0, 0.010, 0.030, 0.060):
            b.link.load(good)
            before = b.link.status().interrupts
            st = b.link.run(good, {"x": x}, trigger_after=delay, load=False)
            out = b.link.outputs(good)
            R, _ = b.link.regs()
            if st.interrupts == before and st.status == S.HALTED \
                    and st.retired == st0.retired:
                # not a core fault: the trigger arrived after HALT
                late += 1
                print(f"  note: trigger {delay * 1000:.0f} ms after G arrived "
                      "after HALT (test timing, not a mismatch)")
                continue
            b.check("catalytic loop: interrupt taken",
                    st.interrupts == before + 1 and st.status == S.HALTED)
            b.check("catalytic loop: acc unchanged by the handler",
                    out["acc"] == acc0, f"{out['acc']} vs {acc0}")
            b.check("catalytic loop: borrowed register restored", R[2] == x)
            b.check("catalytic loop: retired = loop + handler",
                    st.retired == st0.retired + 7)
            landed.append(st.disp_retired)
        mid = [k for k in landed if 0 < k < st0.retired]
        b.check("catalytic loop: some interrupts landed mid-loop", bool(mid),
                f"dispatch points {landed} of {st0.retired}")
        # the non-restoring handler: the verifier rejects it, and on the
        # board the loop's result changes when it lands mid-loop
        b.link.load(bad)
        st = b.link.run(bad, {"x": x}, trigger_after=0.010, load=False)
        out = b.link.outputs(bad)
        changed = out["acc"] != acc0
        print(f"{'catalytic loop x=' + str(x):28} {st0.retired} instructions, "
              f"T landed at {landed}; bad handler at {st.disp_retired}: acc "
              f"{'changed' if changed else 'unchanged'} ({out['acc']} vs {acc0})")
    print(f"{'interrupts':28}{' OK' if b.fails == fails else ' FAILED'}"
          f"{f' ({late} triggers too late to count)' if late else ''}")


def projection(b: Board, rng: random.Random, quick: bool) -> None:
    """Roots-of-unity projection programs (`cmtz/isa/projection.py`): the
    board against the simulator (whole state) and the oracle."""
    fails = b.fails
    gates = list(itertools.product(range(16), (0, 1), (0, 1)))
    if quick:
        gates = [(tt, rng.randrange(2), rng.randrange(2)) for tt in range(16)]
    for fn in (PR.gate_direct, PR.gate_projection, PR.gate_clean):
        demo = fn()
        f0 = b.fails
        for tt, x, z in gates:
            vals = {**PR.gate_inputs(tt, x, z),
                    **{k: rng.randrange(CORE.p) for k in ("t1", "t2", "s", "y")}}
            vals = {k: v for k, v in vals.items() if k in demo.image.inputs}
            st, m = b.run(demo.image, f"{fn.__name__} {tt} {x}{z}", vals)
            out = b.link.outputs(demo.image)
            want = {"y": (vals["y"] + gate(tt, x, z)) % CORE.p}
            want.update({k: vals[k] for k in demo.layout.catalytic})
            b.check(f"{fn.__name__} {tt} {x}{z}: oracle", out == want, f"{out}")
        print(f"{fn.__name__:28} {len(gates)} gate/input runs, {m.retired} "
              f"instructions, {m.cycles} cycles each"
              f"{' OK' if b.fails == f0 else ' FAILED'}")

    demo, f0 = PR.gate_sweep(), b.fails
    sweep = gates[::4] if quick else gates
    for n, (tt, x, z) in enumerate(sweep):
        y0 = rng.randrange(CORE.p)
        st, m = b.run(demo.image, f"gate_sweep {tt} {x}{z}",
                      {**PR.gate_inputs(tt, x, z), "y0": y0,
                       "s": rng.randrange(CORE.p)})
        got = b.link.read_data(0, 3 * CORE.p * CORE.p)
        want = PR.sweep_expected(tt, x, z, y0)
        b.check(f"gate_sweep {tt} {x}{z}: oracle", got == want,
                f"{sum(g != w for g, w in zip(got, want))} words differ")
    print(f"{'gate_sweep (all 289 masks)':28} {len(sweep)} gate/input runs"
          f"{' OK' if b.fails == f0 else ' FAILED'}")

    for fn in (PR.tree_direct, PR.tree_clean):
        demo, f0 = fn(), b.fails
        n = 4 if quick else 16
        for _ in range(n):
            tts = [rng.randrange(16) for _ in range(3)]
            ins = [rng.randrange(2) for _ in range(4)]
            regs = {k: rng.randrange(CORE.p) for k in ("t1", "t2", "tl", "tr", "s", "y")}
            vals = {**PR.tree_inputs(*tts, *ins),
                    **{k: v for k, v in regs.items() if k in demo.image.inputs}}
            st, m = b.run(demo.image, f"{fn.__name__} {tts} {ins}", vals)
            out = b.link.outputs(demo.image)
            want = {k: v for k, v in gate_tree(*tts, *ins, regs).items()
                    if k in out}
            b.check(f"{fn.__name__} {tts} {ins}: oracle", out == want, f"{out}")
        print(f"{fn.__name__:28} {n} random trees, {m.retired} instructions, "
              f"{m.cycles} cycles each{' OK' if b.fails == f0 else ' FAILED'}")

    # a handler that borrows mask t1 while gate_clean uses it; as in
    # interrupts(), main has to outlast the ~6 ms before T can arrive
    outer, inner = (26, 200) if quick else (52, 200)
    good = PR.gate_irq(outer, inner)
    bad = PR.gate_irq(outer, inner, restore=False)
    f0 = b.fails
    for tt in (PR.GATES["AND"], PR.GATES["XOR"]):
        vals = {**PR.gate_inputs(tt, 1, 1),
                **{k: rng.randrange(CORE.p) for k in ("t1", "t2", "s", "y")}}
        st0, m0 = b.run(good.image, "gate_irq, no T", vals)
        b.check("gate_irq: y gains outer*inner*g",
                m0.outputs()["y"] == (vals["y"] + outer * inner * gate(tt, 1, 1)) % CORE.p)
        landed = []
        for delay in (0.0, 0.010, 0.030, 0.060):
            b.link.load(good.image)
            before = b.link.status().interrupts
            st = b.link.run(good.image, vals, trigger_after=delay, load=False)
            out = b.link.outputs(good.image)
            if st.interrupts == before:
                print(f"  note: trigger {delay * 1000:.0f} ms after G arrived after HALT")
                continue
            b.check("gate_irq: y and masks unchanged by the handler",
                    out == m0.outputs(), f"{out} vs {m0.outputs()}")
            b.check("gate_irq: retired = run + handler", st.retired == st0.retired + 7)
            landed.append(st.disp_retired)
        b.check("gate_irq: interrupts landed mid-run",
                any(0 < k < st0.retired for k in landed), f"{landed}")
        b.link.load(bad.image)
        st = b.link.run(bad.image, vals, trigger_after=0.010, load=False)
        out = b.link.outputs(bad.image)
        print(f"{'gate_irq ' + PR.gate_name(tt):28} {st0.retired} instructions, T "
              f"landed at {landed}; non-restoring handler at {st.disp_retired}: "
              f"{'changed ' + str(out) if out != m0.outputs() else 'unchanged'}")
    print(f"{'projection':28}{' OK' if b.fails == fails else ' FAILED'}")


def soundness(b: Board) -> None:
    """Values outside F_p on the board. These are counterexamples to the
    unguarded contract, not checks of a proof: each must show exactly what
    the RTL simulation shows (`test_isa_catalytic.TestRtlCounterexamples`).
    The verifier now rejects the handler, and the host refuses inputs in
    [p, q); here they are written raw on purpose."""
    fails = b.fails
    img = noncanonical_handler(2000, 300)
    st = b.link.run(img, {"x": 0})
    b.check("noncanonical handler, no T: x stays 0",
            b.link.outputs(img)["x"] == 0)
    b.link.load(img)
    st = b.link.run(img, {"x": 0}, trigger_after=0.010, load=False)
    x = b.link.outputs(img)["x"]
    b.check("noncanonical handler: T leaves x = 13, as the RTL predicts",
            st.interrupts and x == 13, f"x = {x}")
    print(f"{'handler reading R7 = 18':28} x = 0 came back as {x} "
          f"(T at retired {st.disp_retired})")
    demo = PR.gate_clean()
    for t1, back in ((17, 0), (18, 1)):
        vals = {**PR.gate_inputs(PR.GATES["AND"], 1, 1), "t1": 0, "t2": 5,
                "s": 9, "y": 3}
        b.link.load(demo.image)
        for k, v in vals.items():
            b.link.write_reg(demo.image.inputs[k], v)
        b.link.write_reg(PR.T1, t1)
        b.link.go()
        b.link.wait()
        out = b.link.outputs(demo.image)
        b.check(f"gate_clean, mask {t1}: comes back as {back}",
                out == {"t1": back, "t2": 5, "s": 9, "y": 4}, f"{out}")
        print(f"{'gate_clean mask t1 = ' + str(t1):28} came back as {out['t1']}, "
              f"y = {out['y']}")
    print(f"{'soundness counterexamples':28}"
          f"{' as predicted' if b.fails == fails else ' NOT as predicted'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SECTION",
                    choices=["examples", "sweeps", "interrupts", "projection",
                             "soundness"],
                    help="run only these sections")
    args = ap.parse_args()
    path = args.port or (glob.glob("/dev/serial/by-id/*if01-port0") or [None])[0]
    if not path:
        print("no UART found (expected /dev/serial/by-id/*-if01-port0)")
        return 2
    port = Port(path)
    port.drain()
    link = CoreLink(Serial(port), CORE)
    link.check_core()
    print(f"cmtz-core F_{CORE.p} found on {path}")
    b = Board(link)
    rng = random.Random(1)
    only = set(args.only or ["examples", "sweeps", "interrupts", "projection",
                             "soundness"])
    if "examples" in only:
        examples(b, rng, 3 if args.quick else 10)
    if "sweeps" in only:
        sweeps(b)
    if "interrupts" in only:
        interrupts(b, args.quick)
    if "projection" in only:
        projection(b, rng, args.quick)
    if "soundness" in only:
        soundness(b)
    print(f"\n{b.checks - b.fails} of {b.checks} checks matched")
    return 0 if b.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

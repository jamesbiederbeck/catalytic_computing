"""Check a multilane cmtz-core bitstream (RFC-003: cmtz-dual, cmtz-lease) on
the board against the multilane model, and measure it.

Usage: uv run python examples/fpga/mlane_run.py [--port ...] [--quick]
           [--only independent interrupts leases throughput] [--json out.json]

Before every scene it reads the whole board state (every lane's registers,
index registers and data memory, and the lease table) and seeds the model
(`cmtz/isa/multicore.py`) with it; then it runs the scene on both and
compares every lane's status, error, retired count, cycle count (from the
on-chip stamps), stall cycles, all registers, index registers, data memory
and lease state. Program outputs are also checked against the plain-integer
oracle (`cmtz/projection_oracle.py`) and, for independent lanes, each lane
against a scalar `Machine` run on its own.

Sections:
  independent  different programs on the lanes, started on the same clock;
               every lane also alone, for the throughput comparison
  interrupts   'T' sent to one lane while both run: its result must not
               depend on where it lands, the other lane must not notice
  leases       (lease builds) gates and gate trees evaluated in another
               lane's live registers while that lane runs; refusals, poison
               and recovery; random leases
  throughput   one batch of jobs on one lane, then split over the lanes

Reads and writes the serial port only; load the bitstream first
(openFPGALoader -c ft2232 _build/default/hardware.fs). Exit status 0 only
if everything matched.
"""

import argparse
import glob
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # uart_check
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # cmtz

from uart_check import Port

from cmtz.backends.isa_mlane import (
    LaneRun, LeaseOp, Scene, run_scene, scene_model, standard_scenes,
)
from cmtz.isa import CoreConfig, Machine
from cmtz.isa import projection as PR
from cmtz.isa import workloads as W
from cmtz.isa.lease import Lease
from cmtz.isa.mlane_host import MLaneLink
from cmtz.isa.multicore import _Active
from cmtz.projection_oracle import gate, gate_tree

CORE = CoreConfig(17, 19)
P = CORE.p
CLOCK_HZ = 27_000_000


class Serial:
    def __init__(self, port: Port):
        self.port = port

    def read(self, n: int) -> bytes:
        return self.port.read(n, timeout=0.5)

    def write(self, data: bytes) -> None:
        self.port.write(data)


class Board:
    def __init__(self, link: MLaneLink):
        self.link = link
        self.n, self.lease = link.lanes, link.lease
        self.fails = self.checks = 0
        self.log: list[dict] = []

    def check(self, what: str, ok: bool, detail: str = "") -> bool:
        self.checks += 1
        if not ok:
            self.fails += 1
            print(f"  MISMATCH {what} {detail}")
        return ok

    def model(self):
        """A model seeded with the whole board state."""
        mc = scene_model(CORE, self.n)
        for lane in range(self.n):
            self.link.select(lane)
            R, X = self.link.regs()
            mc.P[lane * 32:(lane + 1) * 32] = R
            mc.lanes[lane].X[:] = X
            mc.lanes[lane].D[:] = self.link.read_data(0, CORE.dmem_size)
        if self.lease:
            # leases the board holds (a poisoned one between the scene that
            # poisons it and the one that clears it) go into the model too
            for lane in range(self.n):
                self.link.select(lane)
                ls = self.link.lease_state()
                if ls.state == 0:
                    continue
                lease = Lease.banked(ls.owner, ls.bank, ls.count)
                phys = tuple(ls.owner * 32 + r for r in lease.regs)
                a = _Active(lane, lease, phys, 0)
                if ls.state == 1:
                    mc.active[lane] = a
                    for i in phys:
                        mc.holder[i] = lane
                else:
                    mc.poison_of[lane] = a
                    mc.poisoned |= set(phys)
        return mc

    def scene(self, sc: Scene, expect: dict | None = None) -> dict:
        """Run `sc` on the board and on the model; compare everything."""
        mc = self.model()
        before = {lane: list(mc.lanes[lane].D) for lane in range(self.n)}
        R0 = list(mc.P)
        X0 = [list(ln.X) for ln in mc.lanes]
        r = run_scene(mc, sc, mc.t + 1)
        L = self.link
        for lane, reg, v in sc.setup:
            L.select(lane)
            L.write_reg(reg, v)
        for op, code in zip(sc.leases, r["codes"]):
            got = L.lease_cmd(op.borrower, op.owner, op.bank, op.count)
            self.check(f"{sc.name}: lease {op} code", got == code,
                       f"board {got:#x}, model {code:#x}")
        for lane, reg, v in sc.writes:
            L.select(lane)
            L.write_reg(reg, v)
        for run in sc.runs:
            L.select(run.lane)
            if run.load:
                L.load(run.image)
            for k, v in run.inputs.items():
                if k in run.image.inputs:
                    L.write_reg(run.image.inputs[k], v)
        out = {"scene": sc.name, "lanes": {}}
        if sc.runs:
            t0 = time.monotonic()
            L.start(sum(1 << run.lane for run in sc.runs))
            st = L.wait_all(sorted({run.lane for run in sc.runs}))
            out["wall_s"] = time.monotonic() - t0
            for lane, s in st.items():
                res = r["results"][lane]
                got = (s.status, s.error, s.retired, s.cycles, s.stalls)
                want = (res.status, res.error, res.retired, res.cycles,
                        res.stalls)
                self.check(f"{sc.name}: lane {lane} status/retired/cycles/stalls",
                           got == want, f"board {got}, model {want}")
                out["lanes"][lane] = {"retired": s.retired, "cycles": s.cycles,
                                      "stalls": s.stalls, "start": s.t_start,
                                      "end": s.t_end, "status": s.status}
            starts = [v["start"] for v in out["lanes"].values()]
            ends = [v["end"] for v in out["lanes"].values()]
            out["makespan"] = max(ends) - min(starts)
            out["retired"] = sum(v["retired"] for v in out["lanes"].values())
        for lane in range(self.n):
            L.select(lane)
            R, X = L.regs()
            D = L.read_data(0, CORE.dmem_size)
            self.check(f"{sc.name}: lane {lane} registers",
                       R == mc.P[lane * 32:(lane + 1) * 32],
                       f"board {R}, model {mc.P[lane * 32:(lane + 1) * 32]}")
            self.check(f"{sc.name}: lane {lane} index registers",
                       X == mc.lanes[lane].X)
            diff = [a for a in range(CORE.dmem_size) if D[a] != mc.lanes[lane].D[a]]
            self.check(f"{sc.name}: lane {lane} data memory", not diff,
                       f"{len(diff)} words differ")
            if self.lease:
                ls = L.lease_state()
                want = (1 if lane in mc.active else
                        2 if lane in mc.poison_of else 0)
                self.check(f"{sc.name}: lane {lane} lease state",
                           ls.state == want, f"board {ls}, model {want}")
        for lane, res in r["results"].items():
            run = next(x for x in sc.runs if x.lane == lane)
            if not run.irq_at and not sc.leases:
                # an independent lane equals a scalar Machine on its own
                m = Machine(CORE, run.image)
                m.R = R0[lane * 32:(lane + 1) * 32]
                m.X, m.D = list(X0[lane]), list(before[lane])
                m.set_inputs({k: v for k, v in run.inputs.items()
                              if k in run.image.inputs})
                m.run(max_steps=10_000_000)
                self.check(f"{sc.name}: lane {lane} = scalar simulator",
                           (m.status, m.retired, m.cycles, m.R)
                           == (res.status, res.retired, res.cycles,
                               mc.P[lane * 32:(lane + 1) * 32]))
        for lane, want in (expect or {}).items():
            res = r["results"][lane]
            self.check(f"{sc.name}: lane {lane} oracle",
                       all(res.outputs.get(k) == v for k, v in want.items()),
                       f"outputs {res.outputs}, oracle {want}")
        self.log.append(out)
        return out


# ── workloads ────────────────────────────────────────────────────────────────

def gate_run(rng, lane, demo=None):
    demo = demo or PR.gate_clean()
    tt, a, b, y = rng.randrange(16), rng.randrange(2), rng.randrange(2), rng.randrange(P)
    ins = {**PR.gate_inputs(tt, a, b), "y": y,
           **{k: rng.randrange(P) for k in ("t1", "t2", "s")}}
    return (LaneRun(lane, demo.image, ins),
            {"y": (y + gate(tt, a, b)) % P})


def tree_run(rng, lane, demo=None):
    demo = demo or PR.tree_clean()
    f, g1, g2 = (rng.randrange(16) for _ in range(3))
    bits = tuple(rng.randrange(2) for _ in range(4))
    ins = {**PR.tree_inputs(f, g1, g2, *bits), "y": rng.randrange(P),
           **{k: rng.randrange(P) for k in ("t1", "t2", "tl", "tr", "s")}}
    want = gate_tree(f, g1, g2, *bits, {k: ins[k] for k in
                                        ("y", "t1", "t2", "tl", "tr", "s")})
    return LaneRun(lane, demo.image, ins), {"y": want["y"]}


def owner_run(rng, lane, n=None):
    n = n or rng.randrange(20, 400)
    x, c = rng.randrange(P), rng.randrange(P)
    img = replace(W.owner_program(n), inputs={"x": 1, "c": 2})
    return LaneRun(lane, img, {"x": x, "c": c}), n, x, c


PROGS = {
    "gate_direct": lambda rng, l: gate_run(rng, l, PR.gate_direct()),
    "gate_projection": lambda rng, l: gate_run(rng, l, PR.gate_projection()),
    "gate_clean": gate_run,
    "tree_direct": lambda rng, l: tree_run(rng, l, PR.tree_direct()),
    "tree_clean": tree_run,
}


def independent(b: Board, rng: random.Random, quick: bool) -> None:
    for sc in standard_scenes(CORE, b.n, False, seed=rng.randrange(1000)):
        if any(r.offset or r.irq_at for r in sc.runs):
            continue
        b.scene(sc)
    print(f"{'standard scenes':28s} OK" if not b.fails else "standard scenes: failures")
    names = list(PROGS)
    pairs = [(x, y) for x in names for y in names]
    if quick:
        pairs = pairs[::4]
    rows = []
    for x, y in pairs:
        runs, exp = [], {}
        for lane, name in enumerate((x, y)[:b.n]):
            run, want = PROGS[name](rng, lane)
            runs.append(run)
            exp[lane] = want
        alone = []
        for run in runs:
            out = b.scene(Scene(f"{x}+{y} lane {run.lane} alone", [run]),
                          {run.lane: exp[run.lane]})
            alone.append(out)
        both = b.scene(Scene(f"{x}+{y} together", runs), exp)
        seq = sum(o["makespan"] for o in alone)
        rows.append({"pair": f"{x}+{y}", "retired": both["retired"],
                     "sequential_cycles": seq, "together_cycles": both["makespan"],
                     "ipc_one_lane": both["retired"] / seq,
                     "ipc_two_lanes": both["retired"] / both["makespan"]})
    b.log.append({"section": "independent", "rows": rows})
    for row in rows:
        print(f"  {row['pair']:34s} IPC one lane {row['ipc_one_lane']:.3f}, "
              f"two lanes {row['ipc_two_lanes']:.3f} "
              f"(x{row['ipc_two_lanes'] / row['ipc_one_lane']:.2f})")
    print(f"{'independent':28s} {len(pairs)} pairs", "OK" if not b.fails else "")


def interrupts(b: Board, rng: random.Random, quick: bool) -> None:
    """Lane 0 runs gate_irq (its handler borrows a mask and restores it)
    for millions of instructions; lane 1 runs tree_clean back to back. 'T'
    goes to lane 0 at a random moment: lane 0's result must equal the
    oracle and its masks come back; lane 1 is compared with the model (it
    cannot notice lane 0's interrupt)."""
    L = b.link
    for restore in (True, False):
        demo = PR.gate_irq(400, 20, restore)
        tt = rng.choice([6, 8, 14])
        a_, b_ = rng.randrange(2), rng.randrange(2)
        ins = {**PR.gate_inputs(tt, a_, b_),
               **{k: rng.randrange(P) for k in ("t1", "t2", "s", "y")}}
        run1, want1 = tree_run(rng, 1)
        mc = b.model()
        r = run_scene(mc, Scene("lane 1 alone (model)", [run1]), mc.t + 1)
        L.select(0)
        L.load(demo.image)
        for k, v in ins.items():
            L.write_reg(demo.image.inputs[k], v)
        L.select(1)
        L.load(run1.image)
        for k, v in run1.inputs.items():
            if k in run1.image.inputs:
                L.write_reg(run1.image.inputs[k], v)
        L.start(3)
        time.sleep(rng.uniform(0.005, 0.08))
        L.select(0)
        L.trigger()
        st = L.wait_all([0, 1])
        L.select(0)
        R, _ = L.regs()
        y = R[demo.image.outputs["y"]]
        n = 400 * 20
        good = y == (ins["y"] + n * gate(tt, a_, b_)) % P
        masks = all(R[demo.image.inputs[k]] == ins[k] for k in ("t1", "t2", "s"))
        landed = st[0].disp_retired
        if restore:
            b.check("interrupt lane 0 result", good and masks,
                    f"y {y}, masks {masks}")
        else:
            b.check("non-restoring handler changes lane 0", not (good and masks))
        b.check("interrupt lane 0 dispatched", st[0].interrupts > 0)
        L.select(1)
        R1, X1 = L.regs()
        res = r["results"][1]
        b.check("interrupt lane 1 untouched",
                (st[1].retired, st[1].cycles, R1) ==
                (res.retired, res.cycles, mc.P[32:64]))
        b.check("interrupt lane 1 oracle", R1[run1.image.outputs["y"]] == want1["y"])
        print(f"{'gate_irq restore=' + str(restore):28s} {st[0].retired} instructions, "
              f"T at {landed}; lane 1 tree_clean {st[1].cycles} cycles unchanged")


def leases(b: Board, rng: random.Random, quick: bool) -> None:
    if not b.lease:
        print("leases: not a lease build, skipped")
        return
    for sc in standard_scenes(CORE, b.n, True, seed=rng.randrange(1000)):
        if any(r.offset or r.irq_at for r in sc.runs):
            continue
        if not sc.leases and not sc.setup and not sc.writes:
            continue
        b.scene(sc)
    print(f"{'standard lease scenes':28s}", "OK" if not b.fails else "failures")
    lg, lt = W.leased(PR.gate_clean()), W.leased(PR.tree_clean())
    stalls, restored = [], 0
    count = 6 if quick else 40
    for i in range(count):
        owner, borrower = rng.sample(range(b.n), 2)
        tree = i % 5 == 4
        nreg = 5 if tree else 3
        bank = rng.choice([1, 2] if tree else [1, 2])
        regs = [bank * 8 + k for k in range(nreg)]
        live = [(owner, r, rng.randrange(P)) for r in regs]
        if tree:
            run, want = tree_run(rng, borrower, None)
            run = LaneRun(borrower, lt.image, {k: v for k, v in run.inputs.items()
                                               if k in lt.image.inputs})
        else:
            run, want = gate_run(rng, borrower)
            run = LaneRun(borrower, lg.image, {k: v for k, v in run.inputs.items()
                                               if k in lg.image.inputs})
        # the owner loops on R1..R3 and then reads one leased register
        n = rng.randrange(10, 300)
        src = (f"LIX X1, 0\nLIX X2, {n}\nl: MUL R3, R1, R2\nADD R1, R3, R2\n"
               f"ADDIX X1, X1, 1\nBLTX X1, X2, l\nADD R5, R{regs[0]}, R1\nHALT")
        from cmtz.isa.asm import assemble
        from cmtz.isa.image import ProgramImage
        a = assemble(src)
        oimg = ProgramImage(P, CORE.q, a.words, a.entry, a.handler,
                            inputs={"x": 1, "c": 2}, outputs={"r": 5})
        orun = LaneRun(owner, oimg, {"x": rng.randrange(P), "c": rng.randrange(P)})
        # the masks now come from the owner's registers; the gain of y does
        # not depend on them, so the oracle's y still holds
        out = b.scene(Scene(f"lease {i}: {'tree' if tree else 'gate'} "
                            f"{borrower}<-{owner} bank {bank}",
                            [run, orun], leases=[LeaseOp(borrower, owner, bank, nreg)],
                            setup=live), {borrower: want})
        # the owner's live values must be back, untouched
        b.link.select(owner)
        R, _ = b.link.regs()
        ok = all(R[r] == v for _, r, v in live)
        b.check(f"lease {i}: owner registers restored", ok)
        restored += ok
        stalls.append(out["lanes"][owner]["stalls"])
    b.log.append({"section": "leases", "random": count, "restored": restored,
                  "owner_stalls": stalls})
    print(f"{'random leases':28s} {count} runs, owner registers restored in "
          f"{restored}; owner stalls {min(stalls)}..{max(stalls)} cycles")


def throughput(b: Board, rng: random.Random, quick: bool) -> None:
    """The same batch of jobs on lane 0 alone, then dealt round-robin to
    every lane; on-chip cycles from the stamps, wall time from the host."""
    batch = [rng.choice(["gate_clean", "gate_projection", "tree_clean"])
             for _ in range(4 if quick else 12)]
    runs = [PROGS[name](rng, 0) for name in batch]
    one = {"cycles": 0, "retired": 0, "wall": 0.0}
    for run, want in runs:
        o = b.scene(Scene("throughput one lane", [run]), {0: want})
        one["cycles"] += o["makespan"]
        one["retired"] += o["retired"]
        one["wall"] += o["wall_s"]
    many = {"cycles": 0, "retired": 0, "wall": 0.0}
    for i in range(0, len(runs), b.n):
        group = runs[i:i + b.n]
        sc = Scene("throughput lanes", [replace(run, lane=k)
                                        for k, (run, _) in enumerate(group)])
        o = b.scene(sc, {k: want for k, (_, want) in enumerate(group)})
        many["cycles"] += o["makespan"]
        many["retired"] += o["retired"]
        many["wall"] += o["wall_s"]
    row = {"jobs": len(batch), "one_lane": one, "lanes": many,
           "speedup_cycles": one["cycles"] / many["cycles"]}
    b.log.append({"section": "throughput", **row})
    print(f"{'throughput':28s} {len(batch)} jobs: one lane {one['cycles']} "
          f"cycles (IPC {one['retired'] / one['cycles']:.3f}), {b.n} lanes "
          f"{many['cycles']} cycles (IPC {many['retired'] / many['cycles']:.3f}), "
          f"x{row['speedup_cycles']:.2f}; wall {one['wall']:.2f} s vs "
          f"{many['wall']:.2f} s (UART-bound)")


SECTIONS = {"independent": independent, "interrupts": interrupts,
            "leases": leases, "throughput": throughput}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="SECTION", choices=SECTIONS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    path = args.port or (glob.glob("/dev/serial/by-id/*if01-port0") or [None])[0]
    if not path:
        print("no UART found (expected /dev/serial/by-id/*-if01-port0)")
        return 2
    port = Port(path)
    port.drain()
    probe = MLaneLink(Serial(port), CORE, 2, False)
    info = probe.info()
    if info[:4] != b"CMTM":
        print(f"not a multilane bitstream: {info[:4]!r}")
        return 2
    lanes, lease = info[5], bool(info[6])
    link = MLaneLink(Serial(port), CORE, lanes, lease)
    link.check_core()
    print(f"multilane cmtz-core F_{CORE.p}: {lanes} lanes, leases "
          f"{'on' if lease else 'off'}, on {path}")
    b = Board(link)
    rng = random.Random(args.seed)
    for name in args.only or SECTIONS:
        SECTIONS[name](b, rng, args.quick)
    print(f"\n{b.checks - b.fails} of {b.checks} checks matched")
    if args.json:
        args.json.write_text(json.dumps({"lanes": lanes, "lease": lease,
                                         "checks": b.checks, "fails": b.fails,
                                         "log": b.log}, indent=1))
    return 0 if b.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

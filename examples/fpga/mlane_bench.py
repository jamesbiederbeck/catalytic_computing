"""Reproducible benchmarks for RFC-003 (phase A5): lanes (hypothesis 1)
and register borrowing at fixed physical storage (hypothesis 2), kept
apart.

Usage: uv run python examples/fpga/mlane_bench.py [--board-json f.json ...]
                                                   [--json out.json]

Everything here is the multilane model (`cmtz/isa/multicore.py`), which
the RTL testbenches and the board runs match cycle for cycle; numbers that
come from the board are read from the JSON files `mlane_run.py --json`
writes and are labelled as such. Synthesis numbers are read from the APIO
projects' place-and-route reports when they exist.

1. Throughput. One batch of jobs (the five projection programs in random
   order) on 1, 2, 4 and 8 lanes, each lane taking the next job as soon as
   it is free. Reports makespan, aggregate IPC, jobs per second at 27 MHz,
   and per-job latency (end minus arrival), with no dispatch cost and with
   the measured UART cost per job.
2. Storage. Two lanes. Lane 0 runs an owner job that keeps five live values
   in R16..R20 while it loops, then reads them; lane 1 runs a stream of
   gates (or gate trees). Lane 1's register budget is lowered until the
   stream no longer runs, per policy; at each policy's smallest budget the
   total storage (both register files, spill words in data memory, lease
   metadata) and the time are reported.
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cmtz.analysis.lease_invariants import check_trace
from cmtz.analysis.lease_proof import Checker
from cmtz.isa import projection as PR
from cmtz.isa import workloads as W
from cmtz.isa.asm import assemble
from cmtz.isa.image import ProgramImage
from cmtz.isa.multicore import Job, LaneConfig, MultiCore, Variant

CORE = PR.CORE
P = CORE.p
HZ = 27_000_000
PROJECTS = Path(__file__).resolve().parents[3] / "playground/fpga/projects/cmtz"


# ── 1. throughput ────────────────────────────────────────────────────────────

def batch(rng: random.Random, n: int) -> list[Job]:
    kinds = [PR.gate_direct, PR.gate_projection, PR.gate_clean,
             PR.tree_direct, PR.tree_clean]
    jobs = []
    for i in range(n):
        demo = rng.choice(kinds)()
        if "g1_k0" in demo.layout.inputs:
            ins = PR.tree_inputs(*[rng.randrange(16) for _ in range(3)],
                                 *[rng.randrange(2) for _ in range(4)])
        else:
            ins = PR.gate_inputs(rng.randrange(16), rng.randrange(2),
                                 rng.randrange(2))
        ins.update({k: rng.randrange(P) for k in demo.image.inputs
                    if k not in ins})
        jobs.append(Job(f"j{i}", [Variant(demo.image)], ins))
    return jobs


def throughput(seed: int, n: int, dispatch: int) -> list[dict]:
    rows = []
    for lanes in (1, 2, 4, 8):
        mc = MultiCore(CORE, lanes, dispatch_cycles=dispatch)
        jobs = batch(random.Random(seed), n)
        mc.submit(*jobs)
        rep = mc.run()
        assert not rep.pending and not check_trace(mc.trace)
        lat = sorted(r.end - 0 for r in rep.results)   # all arrive at 0
        rows.append({
            "lanes": lanes, "dispatch_cycles": dispatch,
            "makespan": rep.makespan, "retired": rep.retired,
            "ipc": round(rep.ipc, 3),
            "jobs_per_s": round(n / (rep.makespan / HZ)),
            "latency_mean_us": round(statistics.mean(lat) / HZ * 1e6, 1),
            "latency_p95_us": round(lat[int(0.95 * (n - 1))] / HZ * 1e6, 1),
        })
    return rows


# ── 2. storage at fixed physical size ────────────────────────────────────────

LIVE = (16, 17, 18, 19, 20)


def owner_image(n: int) -> ProgramImage:
    """Loops n times (n = 1000 k) on R1..R3, then reads its five live
    values."""
    assert n % 1000 == 0
    src = f"""
        LIX   X4, 0
        LIX   X5, {n // 1000}
        LIX   X2, 1000
outer:  LIX   X1, 0
loop:   MUL   R3, R1, R2
        ADD   R1, R3, R2
        ADDIX X1, X1, 1
        BLTX  X1, X2, loop
        ADDIX X4, X4, 1
        BLTX  X4, X5, outer
        ADD   R4, R16, R17
        ADD   R4, R4, R18
        ADD   R4, R4, R19
        ADD   R4, R4, R20
        ADD   R5, R1, R4
        HALT
"""
    a = assemble(src)
    return ProgramImage(P, CORE.q, a.words, a.entry, a.handler,
                        inputs={"x": 1, "c": 2}, outputs={"r": 5},
                        listing=[str(i) for i in a.instrs])


def owner_expect(n, x, c, h):
    for _ in range(n):
        x = (x * c + c) % P
    return (x + sum(h)) % P


POLICY_FORMS = {
    # policy name -> (scheduler policy, forms offered)
    "private, direct": ("private", ("direct",)),
    "private, clean": ("private", ("clean",)),
    "private, spill": ("private", ("spill",)),
    "lease (static)": ("static", ("leased",)),
    "least cost (all forms)": ("cost", ("direct", "clean", "spill", "leased")),
}


def storage_run(tree: bool, policy: str, forms: tuple, budget: int,
                jobs: int, seed: int, owner_n: int) -> dict | None:
    rng = random.Random(seed)
    cfgs = [LaneConfig(private=20, lendable=LIVE), LaneConfig(private=budget)]
    mc = MultiCore(CORE, 2, cfgs, policy=policy, snapshot=True,
                   accept=Checker())
    mc.scramble(seed, below=P)
    h = [rng.randrange(P) for _ in LIVE]
    for r, v in zip(LIVE, h):
        mc.write_reg(0, r, v)
    x, c = rng.randrange(P), rng.randrange(P)
    own = Job("owner", [Variant(owner_image(owner_n))], {"x": x, "c": c},
              lane=0, expect={"r": owner_expect(owner_n, x, c, h)})
    stream = []
    for i in range(jobs):
        if tree:
            spec = tuple(rng.randrange(16) for _ in range(3))
            bits = tuple(rng.randrange(2) for _ in range(4))
        else:
            spec, bits = rng.randrange(16), (rng.randrange(2), rng.randrange(2))
        job = W.pressure_job(f"s{i}", "", spec, bits, rng.randrange(P),
                             forms, lane=1)
        if policy == "static":
            # the planned lease: lane 0's live registers, as many as needed
            for v in job.variants:
                if v.window:
                    from cmtz.isa.lease import Lease
                    v.lease = Lease(0, LIVE[:v.window])
                    v.window = 0
        stream.append(job)
    mc.submit(own, *stream)
    rep = mc.run()
    if rep.pending or any(r.status != 2 for r in rep.results):
        return None
    errs = check_trace(mc.trace)
    assert not errs, errs[:3]
    res = rep.by_name()
    assert res["owner"].outputs["r"] == own.expect["r"]
    for j in stream:
        assert res[j.name].outputs["y"] == j.expect["y"], j.name
    s = [res[j.name] for j in stream]
    kinds = sorted({r.kind for r in s})
    vs = W.tree_variants() if tree else W.gate_variants()
    spill = max((vs[k][2] for k in kinds), default=0)
    window = max((vs[k][1] for k in kinds), default=0)
    st = mc.storage()
    return {
        "budget": budget, "kinds": kinds,
        "register_bits": (20 + budget) * CORE.rw,
        "spill_bits": spill * CORE.rw,
        "lease_metadata_bits": st["lease_metadata"] if window else 0,
        "snapshot_bits_debug": rep.snapshot_bits,
        "borrowed": window,
        "stream_cycles": max(r.end for r in s) - min(r.start for r in s),
        "per_job_cycles": round(statistics.mean(r.cycles for r in s), 1),
        "owner_cycles": res["owner"].cycles,
        "owner_stalls": res["owner"].stalls,
        "makespan": rep.makespan,
        "leases": rep.leases["acquire"],
        "restored": rep.leases["restored"],
    }


def storage(tree: bool, seed: int) -> list[dict]:
    jobs, owner_n = (3, 20000) if tree else (12, 3000)
    lo, hi = (14, 24) if tree else (5, 12)
    rows = []
    for name, (policy, forms) in POLICY_FORMS.items():
        best = None
        for budget in range(hi, lo - 1, -1):
            r = storage_run(tree, policy, forms, budget, jobs, seed, owner_n)
            if r is None:
                break
            best = r
        rows.append({"policy": name, **(best or {"budget": None})})
    return rows


# ── synthesis ────────────────────────────────────────────────────────────────

def synthesis() -> list[dict]:
    rows = []
    for p in ("cmtz-core", "cmtz-dual", "cmtz-lease", "cmtz-quad",
              "cmtz-quad-lease"):
        f = PROJECTS / p / "_build/default/hardware.pnr"
        if not f.exists():
            rows.append({"project": p, "built": False})
            continue
        d = json.loads(f.read_text())
        u = d["utilization"]
        rows.append({"project": p, "built": True,
                     **{k: u[k]["used"] for k in ("LUT4", "DFF", "BSRAM")},
                     "fmax_mhz": round(next(iter(d["fmax"].values()))["achieved"], 1)})
    return rows


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--board-json", type=Path, nargs="*", default=[])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {}

    # the UART cost of one job on the board: load, inputs, G, poll, read
    # back; measured by mlane_run.py as wall time per scene (~6 ms)
    uart = 6e-3
    out["throughput_ideal"] = throughput(args.seed, args.jobs, 0)
    out["throughput_uart"] = throughput(args.seed, args.jobs, int(uart * HZ))
    cols = ["lanes", "makespan", "ipc", "jobs_per_s", "latency_mean_us",
            "latency_p95_us"]
    print("## 1. Throughput (model; the board matches it cycle for cycle)\n")
    print(f"{args.jobs} jobs, random mix of the five projection programs, "
          "all queued at cycle 0; a lane takes the next job when free.\n")
    print("No dispatch cost:\n")
    print(table(out["throughput_ideal"], cols))
    print(f"\nWith {uart * 1e3:.0f} ms of host dispatch per job (one UART "
          "round of load, inputs, start and polling):\n")
    print(table(out["throughput_uart"], cols))

    for path in args.board_json:
        d = json.loads(path.read_text())
        for e in d["log"]:
            if e.get("section") == "throughput":
                print(f"\nBoard ({path.name}, {d['lanes']} lanes): {e['jobs']} jobs, "
                      f"one lane {e['one_lane']['cycles']} cycles, all lanes "
                      f"{e['lanes']['cycles']} cycles (x{e['speedup_cycles']:.2f}); "
                      f"wall {e['one_lane']['wall']:.2f} s vs {e['lanes']['wall']:.2f} s")

    cols = ["policy", "budget", "kinds", "register_bits", "spill_bits",
            "lease_metadata_bits", "borrowed", "per_job_cycles",
            "stream_cycles", "owner_stalls", "makespan"]
    for tree in (False, True):
        key = "storage_tree" if tree else "storage_gate"
        out[key] = storage(tree, args.seed)
        print(f"\n## 2. Storage, {'gate tree' if tree else 'gate'} stream "
              "(model)\n")
        print("Lane 0: 20 registers, five live values in R16..R20 it reads "
              "after a long loop. Lane 1: the smallest register budget each "
              "policy runs the stream in.\n")
        print(table(out[key], cols))

    out["synthesis"] = synthesis()
    print("\n## Synthesis (place and route reports)\n")
    print(table(out["synthesis"], ["project", "built", "LUT4", "DFF", "BSRAM",
                                   "fmax_mhz"]))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

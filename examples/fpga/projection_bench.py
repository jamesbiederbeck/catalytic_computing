"""Benchmark the roots-of-unity projection programs on cmtz-core.

Usage: uv run python examples/fpga/projection_bench.py [--board [--port PATH]]

Prints, per program: program words, instructions and clock cycles (exact
and the same for every input, from the clean-computation proof, which is
checked against the simulator; the cycle model is checked against the RTL
in test_isa_core.py), time at 27 MHz, and field registers by role. Over
the UART only the instruction count can be measured (the 'Q' frame): with
--board each program runs on the chip and its retired count is compared.
Wall-clock time over the UART is the ~6 ms round trip, not the core, so it
is not reported.
"""

import argparse
import glob
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # uart_check
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # cmtz

from cmtz.isa import projection as PR

COLS = [("program", "program"), ("words", "words"),
        ("instructions", "instr"), ("cycles", "cycles"),
        ("us_at_27MHz", "µs @27 MHz"), ("inputs", "in"),
        ("catalytic", "catalytic"), ("outputs", "out"), ("temps", "temps"),
        ("index", "X"), ("field_regs", "R total")]


def table(rows: list[dict]) -> str:
    head = "| " + " | ".join(h for _, h in COLS) + " |"
    sep = "|" + "|".join("---" for _ in COLS) + "|"
    body = ["| " + " | ".join(str(r[k]) for k, _ in COLS) + " |" for r in rows]
    base = {r["program"].split("_")[0]: r["instructions"]
            for r in rows if r["program"].endswith("direct")}
    ratio = [f"{r['program']}: {r['instructions'] / base[r['program'].split('_')[0]]:.1f}x"
             for r in rows]
    return "\n".join([head, sep, *body, "", "instructions vs direct: "
                      + ", ".join(ratio)])


def board(rows: list[dict], port: str | None) -> int:
    from core_run import CORE, Serial
    from uart_check import Port
    from cmtz.isa.host import CoreLink
    path = port or (glob.glob("/dev/serial/by-id/*if01-port0") or [None])[0]
    if not path:
        print("no UART found")
        return 2
    p = Port(path)
    p.drain()
    link = CoreLink(Serial(p), CORE)
    link.check_core()
    rng, bad = random.Random(0), 0
    for fn, row in zip(PR.BENCH, rows):
        demo = fn()
        seen = set()
        for _ in range(5):
            vals = {k: rng.randrange(CORE.p) for k in demo.image.inputs}
            seen.add(link.run(demo.image, vals).retired)
        ok = seen == {row["instructions"]}
        bad += not ok
        print(f"board {row['program']:16} retired {sorted(seen)} "
              f"{'= model' if ok else '!= model ' + str(row['instructions'])}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", action="store_true")
    ap.add_argument("--port")
    args = ap.parse_args()
    rows = PR.benchmark()
    print(table(rows))
    return board(rows, args.port) if args.board else 0


if __name__ == "__main__":
    sys.exit(main())

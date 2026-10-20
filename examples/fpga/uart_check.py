"""Check a Cook-Mertz design on the board over its UART, from the terminal.

Usage: uv run python examples/fpga/uart_check.py <example.cmtz> [--basic]
           [--port /dev/serial/by-id/...-if01-port0] [--max 64]

Does what the visualizer's Board panel does, without a browser: compiles the
example, asks the board for its info frame and compares the program hash
(debug protocol), sends vectors, and compares every reply byte, and with the
debug protocol every state byte, with the reference evaluator. Exit status is
0 only if everything matched. Reads and writes the serial port only; it does
not program the board.
"""

import argparse
import glob
import os
import select
import sys
import termios
import time
from pathlib import Path

from cmtz.compiler import compile_program
from cmtz.hwview import HardwareView
from cmtz.backends.verilog_uart import CMD_INFO, CMD_RUN, CMD_STATE, INFO_LEN

EXAMPLES = Path(__file__).resolve().parent.parent


class Port:
    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = termios.B115200
        termios.tcsetattr(self.fd, termios.TCSANOW, a)

    def drain(self, quiet: float = 0.3) -> int:
        """Discard stale bytes (the bridge buffers earlier output)."""
        n = 0
        while select.select([self.fd], [], [], quiet)[0]:
            n += len(os.read(self.fd, 4096))
        return n

    def read(self, n: int, timeout: float = 1.0) -> bytes:
        out, end = b"", time.time() + timeout
        while len(out) < n and time.time() < end:
            if select.select([self.fd], [], [], 0.05)[0]:
                out += os.read(self.fd, n - len(out))
        return out

    def write(self, data) -> None:
        os.write(self.fd, bytes(data))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("example")
    ap.add_argument("--basic", action="store_true",
                    help="projects/cmtz/ builds: no command byte, no state")
    ap.add_argument("--port")
    ap.add_argument("--max", type=int, default=64)
    args = ap.parse_args()

    src = Path(args.example)
    if not src.exists():
        src = EXAMPLES / args.example
    result = compile_program(src.read_text())
    if not result.ok:
        print("compile failed:", result.errors)
        return 2
    hv = HardwareView(result.ir_program)
    m = hv.manifest()
    ni, no, nd = m["request_bytes"], m["reply_bytes"], m["state_bytes"]

    path = args.port or (glob.glob("/dev/serial/by-id/*if01-port0") or [None])[0]
    if not path:
        print("no UART found (expected /dev/serial/by-id/*-if01-port0)")
        return 2
    port = Port(path)
    print(f"port {Path(path).name}; discarded {port.drain()} stale byte(s)")

    ok = True
    if not args.basic:
        port.write([CMD_INFO])
        f = port.read(INFO_LEN)
        if len(f) != INFO_LEN or f[:4] != b"CMTZ":
            print(f"no info frame (got {f.hex(' ') or 'nothing'}); wrong design loaded or wrong protocol?")
            return 1
        h = int.from_bytes(f[10:], "big")
        same = h == m["hash"]
        print(f"info: v{f[4]} request {f[5]} reply {f[6]} state {f[7]} p {f[8]} "
              f"latency {f[9]} hash {h:08x} -> "
              + ("matches this program" if same else f"DIFFERS from compiled {m['hash_hex']}"))
        ok &= same

    vectors = [[0] * ni, [m["p"] - 1] * ni]
    seed = 1
    while len(vectors) < args.max:
        seed = (seed * 1664525 + 1013904223) % 2**32
        v = []
        for _ in range(ni):
            seed = (seed * 1664525 + 1013904223) % 2**32
            v.append(seed % m["p"])
        vectors.append(v)

    bad = 0
    for lanes in vectors:
        want = hv.evaluate(lanes)
        port.write(([CMD_RUN] if not args.basic else []) + lanes)
        reply = list(port.read(no))
        why = ""
        if reply != want["reply"]:
            why = f"reply {reply} expected {want['reply']}"
        elif not args.basic:
            port.write([CMD_STATE])
            st = list(port.read(nd))
            if st != want["state"]:
                k = next((i for i in range(min(len(st), nd)) if st[i] != want["state"][i]), len(st))
                why = f"state {st} expected {want['state']} (first difference at byte {k})"
        if why:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH {lanes}: {why}")
            port.drain(0.3)
    print(f"{len(vectors)} vectors, {bad} mismatch(es)" + ("" if args.basic else ", state checked"))
    ok &= bad == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

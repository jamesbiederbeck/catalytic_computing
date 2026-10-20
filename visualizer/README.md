# Cook-Mertz IR Visualizer

A single-page web app that compiles `.cmtz` source in real time and renders the IR DAG.

**Source:** https://github.com/jamesbiederbeck/catalytic_computing

## Start

Run from the repo root:

```bash
uv run python visualizer/server.py
```

Then open:

```
http://localhost:8765
```

The server also binds on `0.0.0.0` so any machine on the LAN can reach it at `http://<your-ip>:8765`.

Use `--port` to pick a different port:

```bash
uv run python visualizer/server.py --port 9000
```

## What it shows

- **IR DAG** — nodes laid out by dependency depth, color-coded by type
- **Node details** — field, parents, cost (t/s/r), and type-specific parameters (phase exponent, rotation j, matpow bounds, catalytic register list, etc.)
- **Register values** — evaluated by the Python reference backend
- **Phase animation** — scrub through all m = p−1 phase offsets to watch register values sweep the root-of-unity orbit
- **Analysis report** — field consistency, cycle check, catalytic verification, cost propagation results
- **Optimization stats** — rotate fusion, CSE, constant folding counts
- **Example picker** — loads any `.cmtz` file from the `examples/` directory

## Board panel (FPGA over Web Serial)

The **⚡ Board** button opens a panel that runs the compiled program on a Tang
Nano 20K over its UART, reads the design's internal state back, and paints it
onto the DAG: each node shows the value the FPGA holds, in red with the
expected value if it differs, and a chip (`port`, `s1`, `s2`, …, `const`)
gives its pipeline stage. Programs the hardware cannot express (`matpow`)
say so in the panel.

1. Serve the page from `localhost` (`http://localhost:8765`). Web Serial needs
   Chrome or Edge and a secure context, so the LAN address will not work.
2. Pick **Web Serial**, press Connect and choose the board's USB serial port
   (the BL616 shows up as an FT2232, `0403:6010`; one of its two channels is
   the UART, the other JTAG). Stale bytes from an earlier design are
   discarded on connect.
3. **Run** sends the input lanes, compares the reply with the reference
   backend, and (debug protocol) reads the state back. **Sweep** repeats that
   for every input combination or a fixed sample. **live** polls the state.

Two protocols, selectable in the panel:

| Protocol | Boards | What works |
|---|---|---|
| debug | builds from `examples/fpga/build_projects.py --debug` (`cmtz-*-debug`) | Verify (program hash), Run, Read state, live, Sweep with state |
| basic | the projects under `projects/cmtz/` | Run and Sweep (outputs only) |

The debug protocol is documented in `cmtz/backends/verilog_uart.py`: `I`
returns a 14-byte info frame carrying a hash of the hardware structure, which
Verify compares with the program compiled here; `R` + request bytes returns
the reply; `S` returns one byte per lane of every input and registered node.

**Simulated board** speaks the same bytes without hardware, with answers from
the reference backend, so the whole panel can be tried (and tested) offline.
Its **corrupt** option flips a bit in one node's state byte to show how a
mismatch looks.

Files: `board.js` (protocol client and the two transports, no DOM),
`board-ui.js` (panel), `board.test.mjs` (Node tests),
`testdata/golden.json` (bytes from the Python encoder, regenerated with
`UPDATE_GOLDEN=1 uv run pytest cmtz/tests/test_hwview.py`).

```bash
node --test visualizer/board.test.mjs          # protocol client
uv run pytest cmtz/tests/test_hwview.py        # hardware view + server routes
```

## API

The server exposes the endpoints used by the SPA:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/examples` | List of `.cmtz` filenames in `examples/` |
| `GET` | `/api/example/<name>` | Source text of a named example |
| `POST` | `/api/compile` | Compile source JSON `{"source": "..."}` → graph JSON |
| `POST` | `/api/hw/eval` | `{"source": "...", "lanes": [...]}` → reference reply, state bytes and every node's value for that request |
| `GET` | `/board.js`, `/board-ui.js` | The board panel's modules |

The `/api/compile` response includes `nodes`, `analysis`, `optimization`, `frames` (one per phase offset), `m`, and `hw`: the hardware manifest (per-node kind and pipeline stage, UART lane layout, program hash) or `{"ok": false, "error": ...}` if the program has no hardware lowering.

## Node colors

| Color | Node type |
|---|---|
| Blue | IREmbed |
| Violet | IRComplexEmbed |
| Green | IRRotate |
| Orange | IRAdd |
| Slate | IRCompose |
| Pink | IRMeasure |
| Cyan | IRPrimitiveRoots |
| Purple | IRConjugate |
| Yellow | IRMagnitudeSq |
| Indigo | IRMatPow |
| Amber | IRCatalyticRegion (interrupt — strict) |
| Gray | IRRegion (catalytic — advisory) |

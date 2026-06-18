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

## API

The server exposes two endpoints used by the SPA:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/examples` | List of `.cmtz` filenames in `examples/` |
| `GET` | `/api/example/<name>` | Source text of a named example |
| `POST` | `/api/compile` | Compile source JSON `{"source": "..."}` → graph JSON |

The `/api/compile` response includes `nodes`, `analysis`, `optimization`, `frames` (one per phase offset), and `m`.

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

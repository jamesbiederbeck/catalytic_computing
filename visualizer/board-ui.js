// Board panel: connect over Web Serial (or a simulated board), run the compiled
// program on the FPGA, read its state back and paint it onto the DAG.
// Loaded as a module after the page's main script, whose top-level names
// (graph, nodeEls, valHue, hslFill, hslStroke, resetNodeColors) it uses.
import {
  Board, SimTransport, WebSerialTransport, TimeoutError, BoardError, hex,
  stateToNodes, replyToNodes, sameValue, fmtValue, sweepVectors,
} from './board.js';

const $ = id => document.getElementById(id);
const panel = $('board-panel');

let g = null;               // last compile result (graph + hw manifest)
let hw = null;              // g.hw if the program has a hardware lowering
let transport = null;
let board = null;
let lastRef = null;         // reference answers for the last request
let busy = false;
let liveTimer = null;
let sweeping = false;

// ── small helpers ────────────────────────────────────────────────────────────

function setMsg(text, cls = '') {
  const m = $('bp-msg'); m.textContent = text; m.className = 'bp-msg ' + cls;
}
function setStatus(text, cls = '') {
  const s = $('bp-status'); s.textContent = text; s.className = 'bp-pill ' + cls;
}
function log(dir, bytes) {
  const pre = $('bp-logtxt');
  const t = new Date().toISOString().slice(11, 23);
  const line = document.createElement('div');
  line.className = dir;
  line.textContent = `${t} ${dir === 'tx' ? '→' : '←'} ${bytes.length ? hex(bytes) : '(nothing)'}`;
  pre.appendChild(line);
  while (pre.childElementCount > 300) pre.firstChild.remove();
  pre.scrollTop = pre.scrollHeight;
}
const laneName = s => s.lane ? `${s.node}.${s.lane}` : s.node;

async function evalRef(lanes) {
  const r = await fetch('/api/hw/eval', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: g._source, lanes }),
  });
  const d = await r.json();
  if (!d.ok) throw new BoardError(d.error);
  return d;
}

function refreshControls() {
  const on = !!board, dbg = on && $('bp-proto').value === 'debug';
  $('bp-connect').textContent = on ? 'Disconnect' : 'Connect';
  $('bp-connect').disabled = !hw && !on;
  $('bp-run').disabled = !on || busy;
  $('bp-state').disabled = !dbg || busy;
  $('bp-verify').disabled = !dbg || busy;
  $('bp-live').disabled = !dbg;
  $('bp-sweep').disabled = !on || (busy && !sweeping);
  $('bp-sweep').textContent = sweeping ? 'Stop' : 'Sweep';
  $('bp-sweep-state').disabled = !dbg;
  $('bp-sim-opts').style.display =
    $('bp-kind').value === 'sim' && $('bp-proto').value === 'debug' ? '' : 'none';
  $('bp-any-wrap').style.display = $('bp-kind').value === 'serial' ? '' : 'none';
}

// ── compile hook ─────────────────────────────────────────────────────────────

function onCompiled(data) {
  g = data;
  hw = data.hw?.ok ? data.hw : null;
  lastRef = null;
  stopLive();
  buildLaneInputs();
  decorateStages();
  buildFaultOptions();
  $('bp-out').innerHTML = '<tr><th>lane</th><th>board</th><th>expected</th></tr>';
  if (!hw) {
    setMsg(`no hardware lowering for this program: ${data.hw?.error ?? 'unknown'}`, 'fail');
  } else {
    setMsg(`${hw.request_bytes} request byte${hw.request_bytes === 1 ? '' : 's'}, ` +
      `${hw.reply_bytes} reply, ${hw.state_bytes} state, latency ${hw.latency}, ` +
      `hash ${hw.hash_hex}`);
  }
  // A new program means a new bitstream for the board: keep the port open,
  // ask it again whether it matches.
  if (board) {
    if (!hw) { disconnect('program has no hardware lowering'); return; }
    makeBoard($('bp-proto').value);
    if ($('bp-proto').value === 'debug') verify();
  }
  refreshControls();
}

function buildLaneInputs() {
  const box = $('bp-lanes');
  const prev = new Map(Array.from(box.querySelectorAll('input'), i => [i.dataset.k, i.value]));
  box.innerHTML = '';
  if (!hw) return;
  const defaults = {};
  for (const n of g.nodes) if (n.value_num != null) defaults[n.name] = n.value_num;
  hw.lanes.inputs.forEach((s, i) => {
    const key = laneName(s);
    const d = defaults[s.node];
    const dv = Array.isArray(d) ? d[s.lane === 'im' ? 1 : 0] : d;
    const lab = document.createElement('label');
    lab.textContent = key; lab.title = `byte ${i} of the request`;
    const inp = document.createElement('input');
    inp.type = 'number'; inp.min = 0; inp.max = hw.p - 1; inp.step = 1;
    inp.dataset.k = key;
    inp.value = prev.get(key) ?? String(dv ?? 1);
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') doRun(); });
    box.append(lab, inp);
  });
}

function readLanes() {
  const vals = Array.from($('bp-lanes').querySelectorAll('input'), i => Number(i.value));
  const bad = vals.findIndex(v => !Number.isInteger(v) || v < 0 || v >= hw.p);
  if (bad >= 0) { setMsg(`lane ${laneName(hw.lanes.inputs[bad])} must be an integer 0..${hw.p - 1}`, 'fail'); return null; }
  return vals;
}

function decorateStages() {
  if (!hw || typeof nodeEls === 'undefined') return;
  for (const n of hw.nodes) {
    const e = nodeEls.get(n.name);
    if (!e) continue;
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    const isConst = n.kind === 'const';
    t.setAttribute('class', 'node-stage' + (isConst ? ' const' : ''));
    t.setAttribute('x', 126); t.setAttribute('y', 12);
    t.textContent = isConst ? 'const' : n.kind === 'input' ? 'port' : `s${n.depth}`;
    const tip = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    tip.textContent = isConst ? 'folded into the hardware at compile time'
      : `${n.kind}, pipeline stage ${n.depth}, ${n.width} bit${n.width === 1 ? '' : 's'}`;
    t.appendChild(tip);
    e.g.appendChild(t);
  }
}

function buildFaultOptions() {
  const sel = $('bp-fault');
  sel.innerHTML = '<option value="">nothing</option>';
  if (!hw) return;
  hw.lanes.state.forEach((s, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = laneName(s); sel.appendChild(o);
  });
}

// ── painting the graph ───────────────────────────────────────────────────────

function paint(fpga, ref) {
  let match = 0, mismatch = 0;
  for (const n of graph.nodes) {
    const e = nodeEls.get(n.name);
    if (!e) continue;
    const v = fpga[n.name];
    const r = ref?.values?.[n.name];
    if (v !== undefined) {
      const bad = r !== undefined && !sameValue(v, r);
      bad ? mismatch++ : match++;
      const hue = valHue(v, hw.p);
      e.rect.setAttribute('fill', hslFill(hue));
      e.rect.setAttribute('stroke', bad ? '#f87171' : hslStroke(hue));
      e.rect.setAttribute('stroke-width', bad ? '3.5' : '2');
      e.valueLabel.setAttribute('fill', bad ? '#f87171' : hslStroke(hue));
      e.valueLabel.textContent = `= ${fmtValue(v)}` + (bad ? ` ≠ ${fmtValue(r)}` : '');
    } else if (r !== undefined) {
      e.valueLabel.setAttribute('fill', '#7880a0');
      e.valueLabel.textContent = `= ${fmtValue(r)}`;
    }
  }
  return { match, mismatch };
}

function clearOverlay() {
  if (!graph) return;
  resetNodeColors();
  nodeEls.forEach(e => e.rect.setAttribute('stroke-width', '1.5'));
}

function renderOutputs(reply, ref) {
  const rows = hw.lanes.outputs.map((s, i) => {
    const ok = reply[i] === ref.reply[i];
    return `<tr><td>${laneName(s)}</td><td class="${ok ? 'ok' : 'bad'}">${reply[i]}</td>` +
           `<td>${ref.reply[i]}</td></tr>`;
  });
  $('bp-out').innerHTML = '<tr><th>lane</th><th>board</th><th>expected</th></tr>' + rows.join('');
}

// ── operations ───────────────────────────────────────────────────────────────

// Run `fn` as the one in-flight operation; report errors in the panel.
async function guarded(fn) {
  if (busy) return;
  busy = true; refreshControls();
  try { await fn(); }
  catch (e) {
    if (e instanceof TimeoutError) {
      setMsg(`no answer from the board (${e.message}). Is this the program loaded, ` +
        `and the right protocol selected?`, 'fail');
      setStatus('no answer', 'fail');
      try { await board?.resync(); } catch {}
    } else {
      setMsg(e.message, 'fail');
    }
    stopLive();
  } finally { busy = false; refreshControls(); }
}

async function doRun() {
  if (!board || !hw) return;
  const lanes = readLanes();
  if (!lanes) return;
  await guarded(async () => {
    setMsg('running…');
    const [ref, reply] = await Promise.all([evalRef(lanes), board.run(lanes)]);
    lastRef = ref;
    renderOutputs(reply, ref);
    const fpga = replyToNodes(hw, reply);
    let text = reply.every((b, i) => b === ref.reply[i])
      ? 'outputs match the reference' : 'OUTPUTS DIFFER from the reference';
    let ok = text.startsWith('outputs');
    if ($('bp-proto').value === 'debug') {
      Object.assign(fpga, stateToNodes(hw, await board.state()));
    }
    const { match, mismatch } = paint(fpga, ref);
    if ($('bp-proto').value === 'debug') {
      text += `; state: ${match} node${match === 1 ? '' : 's'} match` +
              (mismatch ? `, ${mismatch} DIFFER` : '');
      ok = ok && mismatch === 0;
    }
    setMsg(text, ok ? 'ok' : 'fail');
  });
}

async function doState() {
  if (!board || !lastRef) { setMsg('run a request first', ''); return; }
  await guarded(async () => {
    const fpga = stateToNodes(hw, await board.state());
    const { match, mismatch } = paint(fpga, lastRef);
    setMsg(`state: ${match} match` + (mismatch ? `, ${mismatch} DIFFER` : ''),
           mismatch ? 'fail' : 'ok');
  });
}

async function verify() {
  await guarded(async () => {
    const v = await board.verify();
    if (v.ok) {
      setStatus('verified', 'ok');
      setMsg(`board matches this program (hash ${hw.hash_hex}, latency ${v.info.latency})`, 'ok');
    } else {
      setStatus('mismatch', 'fail');
      setMsg(`board does not match: ${v.problems.join('; ')}`, 'fail');
    }
  });
}

async function doSweep() {
  if (sweeping) { sweeping = false; return; }
  if (!board || !hw) return;
  const debug = $('bp-proto').value === 'debug' && $('bp-sweep-state').checked;
  const max = Math.max(1, Math.min(4096, Number($('bp-sweep-max').value) || 256));
  const vectors = sweepVectors(hw.p, hw.request_bytes, { max });
  const prog = $('bp-prog');
  prog.max = vectors.length; prog.value = 0;
  sweeping = true;
  const bad = [];
  let done = 0;
  await guarded(async () => {
    refreshControls();
    for (const lanes of vectors) {
      if (!sweeping) break;
      const [ref, reply] = await Promise.all([evalRef(lanes), board.run(lanes)]);
      let why = reply.every((b, i) => b === ref.reply[i]) ? '' : `reply ${reply} vs ${ref.reply}`;
      if (!why && debug) {
        const st = await board.state();
        const k = st.findIndex((b, i) => b !== ref.state[i]);
        if (k >= 0) why = `state ${laneName(hw.lanes.state[k])} = ${st[k]} vs ${ref.state[k]}`;
      }
      if (why) bad.push(`[${lanes}] ${why}`);
      prog.value = ++done;
      if (done % 8 === 0) $('bp-sweep-msg').textContent = `${done}/${vectors.length}, ${bad.length} mismatch(es)`;
    }
  });
  const stopped = !sweeping;
  sweeping = false;
  const m = $('bp-sweep-msg');
  m.className = 'bp-msg ' + (bad.length ? 'fail' : done === vectors.length ? 'ok' : '');
  m.textContent = `${done}/${vectors.length} vector${vectors.length === 1 ? '' : 's'}` +
    `${stopped && done < vectors.length ? ' (stopped)' : ''}: ` +
    (bad.length ? `${bad.length} mismatch(es). First: ${bad.slice(0, 3).join(' | ')}` : 'all match');
  refreshControls();
}

// ── connection ───────────────────────────────────────────────────────────────

function makeBoard(proto) {
  if ($('bp-kind').value === 'sim') {
    transport?.close?.();
    transport = new SimTransport(hw, lanes => evalRef(lanes), { protocol: proto });
    const f = $('bp-fault').value;
    transport.fault = f === '' ? null : Number(f);
  }
  board = new Board(transport, hw, { protocol: proto, onTraffic: log });
}

async function connect() {
  if (board) { await disconnect(); return; }
  if (!hw) return;
  const kind = $('bp-kind').value, proto = $('bp-proto').value;
  try {
    if (kind === 'serial') {
      if (!WebSerialTransport.supported()) {
        setMsg('Web Serial needs Chrome or Edge, and the page served from localhost ' +
          '(http://localhost:8765), not a LAN address.', 'fail');
        return;
      }
      transport = new WebSerialTransport();
      await transport.connect({ anyPort: $('bp-any').checked });
    }
    makeBoard(proto);
    setStatus('connected', 'ok');
    setMsg(kind === 'serial' ? 'port open at 115200 8N1' : 'simulated board ready');
    refreshControls();
    if (proto === 'debug') await verify();
  } catch (e) {
    transport = null; board = null;
    setStatus('offline');
    setMsg(e.name === 'NotFoundError' ? 'no port chosen' : `could not connect: ${e.message}`, 'fail');
  }
  refreshControls();
}

async function disconnect(why = '') {
  stopLive();
  sweeping = false;
  try { await transport?.close(); } catch {}
  transport = null; board = null;
  setStatus('offline');
  if (why) setMsg(why, 'fail');
  refreshControls();
}

// ── live polling ─────────────────────────────────────────────────────────────

function stopLive() {
  clearInterval(liveTimer); liveTimer = null;
  $('bp-live').checked = false;
}
function toggleLive() {
  clearInterval(liveTimer); liveTimer = null;
  if (!$('bp-live').checked) return;
  liveTimer = setInterval(() => { if (!busy && lastRef) doState(); }, 500);
}

// ── wiring ───────────────────────────────────────────────────────────────────

$('board-toggle').addEventListener('click', () => panel.classList.toggle('hidden'));
$('bp-connect').addEventListener('click', connect);
$('bp-verify').addEventListener('click', verify);
$('bp-run').addEventListener('click', doRun);
$('bp-state').addEventListener('click', doState);
$('bp-sweep').addEventListener('click', doSweep);
$('bp-clear').addEventListener('click', clearOverlay);
$('bp-live').addEventListener('change', toggleLive);
$('bp-fault').addEventListener('change', () => {
  if (transport instanceof SimTransport) {
    const f = $('bp-fault').value;
    transport.fault = f === '' ? null : Number(f);
  }
});
for (const id of ['bp-kind', 'bp-proto'])
  $(id).addEventListener('change', async () => { if (board) await disconnect(); refreshControls(); });
window.addEventListener('cmtz-compiled', e => onCompiled(e.detail));

if (!WebSerialTransport.supported()) {
  $('bp-kind').options[0].textContent = 'Web Serial (needs Chrome/Edge on localhost)';
}
refreshControls();
// the first compile may have finished before this module loaded
if (typeof graph !== 'undefined' && graph?.hw) onCompiled(graph);

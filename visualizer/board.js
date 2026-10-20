// UART protocol client for the Cook-Mertz FPGA projects, plus two transports:
// Web Serial (the real board) and a simulated board that speaks the same bytes.
// No DOM here, so Node can test it (see board.test.mjs).
//
// Protocols (cmtz/backends/verilog_uart.py):
//   basic  request = one byte per input lane, reply = one byte per output lane.
//          This is what the projects under projects/cmtz/ speak.
//   debug  adds a command byte: 'I' info frame, 'R' + request -> reply,
//          'S' -> state bytes (one per lane of every input and registered node).
//
// `hw` below is the manifest from /api/compile (cmtz/hwview.py).

export const CMD = { RUN: 0x52, STATE: 0x53, INFO: 0x49 };
export const INFO_LEN = 14;
export const ERROR_BYTE = 0xff;

export class TimeoutError extends Error {
  constructor(want, got) {
    super(`timed out: wanted ${want} byte${want === 1 ? '' : 's'}, got ${got}`);
    this.name = 'TimeoutError';
    this.got = got;
  }
}

export const hex = bytes =>
  Array.from(bytes, b => b.toString(16).padStart(2, '0')).join(' ');

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Byte queue shared by both transports ─────────────────────────────────────

class ByteQueue {
  constructor() { this.bytes = []; this.waiters = []; }
  push(chunk) {
    for (const b of chunk) this.bytes.push(b);
    this.waiters.splice(0).forEach(w => w());
  }
  async take(n, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    while (this.bytes.length < n) {
      const left = deadline - Date.now();
      if (left <= 0) {
        const got = this.bytes.length;
        this.bytes.length = 0;           // a short read leaves the stream unframed
        throw new TimeoutError(n, got);
      }
      await new Promise(res => {
        const t = setTimeout(res, left);
        this.waiters.push(() => { clearTimeout(t); res(); });
      });
    }
    return Uint8Array.from(this.bytes.splice(0, n));
  }
  clear() { this.bytes.length = 0; }
}

// ── Web Serial transport ─────────────────────────────────────────────────────

// The on-board BL616 shows up as an FT2232 (0403:6010); one of its two
// channels is the UART, the other JTAG.
export const BOARD_FILTERS = [{ usbVendorId: 0x0403, usbProductId: 0x6010 }];

export class WebSerialTransport {
  static supported() {
    return typeof navigator !== 'undefined' && 'serial' in navigator;
  }
  constructor() { this.q = new ByteQueue(); this.port = null; }

  async connect({ baudRate = 115200, anyPort = false } = {}) {
    this.port = await navigator.serial.requestPort(
      anyPort ? {} : { filters: BOARD_FILTERS });
    await this.port.open({ baudRate });
    this.writer = this.port.writable.getWriter();
    this.reader = this.port.readable.getReader();
    this._pump();
    // The bridge buffers whatever an earlier design printed; throw it away.
    await this.drain(250);
  }
  async _pump() {
    try {
      for (;;) {
        const { value, done } = await this.reader.read();
        if (done) break;
        if (value) this.q.push(value);
      }
    } catch (e) { this.error = e; }
  }
  async write(bytes) { await this.writer.write(Uint8Array.from(bytes)); }
  read(n, timeoutMs) { return this.q.take(n, timeoutMs); }
  async drain(quietMs = 100) {
    let seen = -1;
    while (this.q.bytes.length !== seen) {
      seen = this.q.bytes.length;
      await sleep(quietMs);
    }
    const n = this.q.bytes.length;
    this.q.clear();
    return n;
  }
  async close() {
    try { await this.reader?.cancel(); } catch {}
    try { this.reader?.releaseLock(); } catch {}
    try { this.writer?.releaseLock(); } catch {}
    try { await this.port?.close(); } catch {}
    this.port = null;
  }
}

// ── Simulated board ──────────────────────────────────────────────────────────

// Speaks the same bytes as the generated Verilog front end, with answers from
// `evaluate(lanes) -> {reply, state}` (the server's reference evaluation).
// Mirrors its framing rules: a stalled request is dropped after
// `stallMs`, other bytes between commands are ignored, replies come back
// whole, and an out-of-range request lane gives all-0xFF.
export class SimTransport {
  constructor(hw, evaluate, { protocol = 'debug', stallMs = 250,
                              fault = null } = {}) {
    this.hw = hw; this.evaluate = evaluate; this.protocol = protocol;
    this.stallMs = stallMs;
    this.fault = fault;                  // state lane index to corrupt, or null
    this.q = new ByteQueue();
    this.mode = protocol === 'debug' ? 'cmd' : 'recv';
    this.req = [];
    this.state = new Array(hw.state_bytes).fill(0);
    this.stale = [];                     // bytes the "bridge" buffered earlier
    this.stallTimer = null;
    this.busy = Promise.resolve();
  }
  async connect() { this.q.push(this.stale); await this.drain(0); }
  async close() { clearTimeout(this.stallTimer); }
  read(n, timeoutMs) { return this.q.take(n, timeoutMs); }
  async drain() { const n = this.q.bytes.length; this.q.clear(); return n; }

  infoFrame() {
    const h = this.hw.hash >>> 0;
    return [0x43, 0x4d, 0x54, 0x5a, 1, this.hw.request_bytes, this.hw.reply_bytes,
            this.hw.state_bytes, this.hw.p, this.hw.latency,
            (h >>> 24) & 255, (h >>> 16) & 255, (h >>> 8) & 255, h & 255];
  }
  async write(bytes) {
    // bytes arrive as they are sent; while a reply is in flight _feed drops them
    for (const b of bytes) this._feed(b);
  }
  _arm() {
    clearTimeout(this.stallTimer);
    this.stallTimer = setTimeout(() => {
      this.req = [];
      this.mode = this.protocol === 'debug' ? 'cmd' : 'recv';
    }, this.stallMs);
  }
  _feed(b) {
    if (this.mode === 'busy') return;     // bytes during a reply are ignored
    if (this.mode === 'cmd') {
      if (b === CMD.INFO) this.q.push(this.infoFrame());
      else if (b === CMD.STATE) {
        const out = this.state.slice();
        if (this.fault != null && this.fault < out.length) out[this.fault] ^= 1;
        this.q.push(out);
      } else if (b === CMD.RUN) { this.mode = 'recv'; this.req = []; this._arm(); }
      return;
    }
    this.req.push(b);
    this._arm();
    if (this.req.length < this.hw.request_bytes) return;
    clearTimeout(this.stallTimer);
    const lanes = this.req; this.req = [];
    this.mode = this.protocol === 'debug' ? 'cmd' : 'recv';
    if (lanes.some(x => x >= this.hw.p)) {
      this.q.push(new Array(this.hw.reply_bytes).fill(ERROR_BYTE));
      return;
    }
    this.mode = 'busy';
    this.busy = this.busy.then(async () => {
      const r = await this.evaluate(lanes);
      this.state = r.state;
      this.q.push(r.reply);
      this.mode = this.protocol === 'debug' ? 'cmd' : 'recv';
    });
  }
}

// ── Board client ─────────────────────────────────────────────────────────────

export class BoardError extends Error {}

export class Board {
  // onTraffic(dir, bytes) with dir 'tx' | 'rx' feeds the serial log.
  constructor(transport, hw, { protocol = 'debug', timeoutMs = 600,
                               onTraffic = () => {} } = {}) {
    this.t = transport; this.hw = hw; this.protocol = protocol;
    this.timeoutMs = timeoutMs; this.onTraffic = onTraffic;
    this.lock = Promise.resolve();
  }
  // Serialise operations: the board has one FSM and one byte stream.
  _exclusive(fn) {
    const run = this.lock.then(fn);
    this.lock = run.catch(() => {});
    return run;
  }
  async _send(bytes) { this.onTraffic('tx', bytes); await this.t.write(bytes); }
  async _recv(n) {
    try {
      const got = await this.t.read(n, this.timeoutMs);
      this.onTraffic('rx', got);
      return got;
    } catch (e) {
      if (e instanceof TimeoutError) this.onTraffic('rx', []);
      throw e;
    }
  }
  _needDebug() {
    if (this.protocol !== 'debug')
      throw new BoardError('this protocol has no such command (basic projects only answer requests)');
  }

  // Info frame -> {version, request, reply, state, p, latency, hash}
  info() {
    this._needDebug();
    return this._exclusive(async () => {
      await this._send([CMD.INFO]);
      const f = await this._recv(INFO_LEN);
      if (String.fromCharCode(...f.slice(0, 4)) !== 'CMTZ')
        throw new BoardError(`not a Cook-Mertz board: ${hex(f.slice(0, 4))}`);
      return {
        version: f[4], request: f[5], reply: f[6], state: f[7], p: f[8],
        latency: f[9],
        hash: ((f[10] << 24) | (f[11] << 16) | (f[12] << 8) | f[13]) >>> 0,
      };
    });
  }

  // Does the design on the board match the program we compiled?
  async verify() {
    const i = await this.info();
    const h = this.hw;
    const problems = [];
    if (i.hash !== (h.hash >>> 0))
      problems.push(`program hash ${i.hash.toString(16).padStart(8, '0')} on the board, ${h.hash_hex} compiled here`);
    if (i.request !== h.request_bytes || i.reply !== h.reply_bytes ||
        i.state !== h.state_bytes || i.p !== h.p)
      problems.push('lane counts or field differ');
    return { ok: problems.length === 0, problems, info: i };
  }

  // One request -> reply lanes. An all-0xFF reply means a lane was out of range.
  run(lanes) {
    if (lanes.length !== this.hw.request_bytes)
      throw new BoardError(`need ${this.hw.request_bytes} lanes, got ${lanes.length}`);
    return this._exclusive(async () => {
      const frame = this.protocol === 'debug' ? [CMD.RUN, ...lanes] : lanes;
      await this._send(frame);
      return Array.from(await this._recv(this.hw.reply_bytes));
    });
  }

  state() {
    this._needDebug();
    return this._exclusive(async () => {
      await this._send([CMD.STATE]);
      return Array.from(await this._recv(this.hw.state_bytes));
    });
  }

  // Get the board back to a known place after a timeout: wait out the
  // stall timeout so a half-received request is dropped, then discard stray bytes.
  async resync() {
    await sleep(300);
    return this.t.drain(50);
  }
}

// ── Decoding ─────────────────────────────────────────────────────────────────

// Lane bytes -> {node: value} where value is an int, or [re, im] if complex.
function groupLanes(slots, bytes) {
  const out = {};
  slots.forEach((s, k) => {
    if (s.lane === '') out[s.node] = bytes[k];
    else (out[s.node] ??= [0, 0])[s.lane === 're' ? 0 : 1] = bytes[k];
  });
  return out;
}
export const stateToNodes = (hw, bytes) => groupLanes(hw.lanes.state, bytes);
export const replyToNodes = (hw, bytes) => groupLanes(hw.lanes.outputs, bytes);

export const sameValue = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export const fmtValue = v => Array.isArray(v) ? `(${v[0]},${v[1]})` : String(v);

// ── Inputs ───────────────────────────────────────────────────────────────────

// Vectors for a sweep: every combination if the space is small, otherwise a
// seeded sample with the corners first. Deterministic, so reruns compare.
export function sweepVectors(p, nLanes, { max = 1024, seed = 1 } = {}) {
  const total = p ** nLanes;
  const out = [];
  if (total <= max) {
    for (let i = 0; i < total; i++) {
      const v = []; let x = i;
      for (let k = 0; k < nLanes; k++) { v.push(x % p); x = Math.floor(x / p); }
      out.push(v);
    }
    return out;
  }
  out.push(new Array(nLanes).fill(0), new Array(nLanes).fill(p - 1));
  let s = seed >>> 0;
  const next = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  while (out.length < max)
    out.push(Array.from({ length: nLanes }, () => Math.floor(next() * p)));
  return out;
}

// Run: node --test visualizer/board.test.mjs
// Replays visualizer/testdata/golden.json, which cmtz/tests/test_hwview.py
// regenerates from the Python side (UPDATE_GOLDEN=1), so the framing here is
// checked against the bytes the generated Verilog expects.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  Board, SimTransport, TimeoutError, BoardError, CMD, INFO_LEN, hex,
  stateToNodes, replyToNodes, sameValue, fmtValue, sweepVectors,
} from './board.js';

const golden = JSON.parse(readFileSync(new URL('./testdata/golden.json', import.meta.url)));
const FIB = golden['fibonacci_pure.cmtz'];
const PHASOR = golden['07_phasor_interference.cmtz'];

// evaluate() backed by the golden cases, like /api/hw/eval would be
const evaluator = g => async lanes => {
  const c = g.cases.find(c => sameValue(c.lanes, lanes));
  if (!c) throw new Error(`no golden case for ${lanes}`);
  return { reply: c.reply, state: c.state, values: c.values };
};
const rig = (g, opts = {}, bopts = {}) => {
  const t = new SimTransport(g.manifest, evaluator(g), opts);
  const log = [];
  const b = new Board(t, g.manifest, { timeoutMs: 100, onTraffic: (d, x) => log.push([d, [...x]]), ...bopts });
  return { t, b, log };
};

test('info frame matches the Python encoder byte for byte', async () => {
  const { t, b } = rig(FIB);
  await t.write([CMD.INFO]);
  const raw = await t.read(INFO_LEN, 100);
  assert.deepEqual([...raw], FIB.info_bytes);
  const v = await b.verify();
  assert.equal(v.ok, true, v.problems.join('; '));
  assert.equal(v.info.latency, 10);
});

test('verify flags a board built from a different program', async () => {
  const other = structuredClone(FIB);
  other.manifest.hash = (FIB.manifest.hash ^ 1) >>> 0;
  const t = new SimTransport(FIB.manifest, evaluator(FIB));
  const b = new Board(t, other.manifest, { timeoutMs: 100 });
  const v = await b.verify();
  assert.equal(v.ok, false);
  assert.match(v.problems[0], /program hash/);
});

for (const [name, g] of Object.entries(golden)) {
  test(`run and state reproduce the reference: ${name}`, async () => {
    const { b } = rig(g);
    for (const c of g.cases) {
      assert.deepEqual(await b.run(c.lanes), c.reply);
      const st = await b.state();
      assert.deepEqual(st, c.state);
      const nodes = stateToNodes(g.manifest, st);
      for (const [node, v] of Object.entries(nodes))
        assert.ok(sameValue(v, c.values[node]), `${node}: ${fmtValue(v)} vs ${fmtValue(c.values[node])}`);
      const outs = replyToNodes(g.manifest, c.reply);
      for (const [node, v] of Object.entries(outs))
        assert.ok(sameValue(v, c.values[node]), node);
    }
  });
}

test('complex lanes decode to [re, im]', async () => {
  const { b } = rig(PHASOR);
  await b.run([6, 3, 6, 3]);
  const nodes = stateToNodes(PHASOR.manifest, await b.state());
  assert.deepEqual(nodes.sum_AB, [5, 6]);
  assert.deepEqual(nodes.cembed_1, [6, 3]);
});

test('an out-of-range lane gives an all-0xFF reply', async () => {
  const { b } = rig(PHASOR);
  assert.deepEqual(await b.run([7, 0, 0, 0]), [255, 255, 255]);
});

test('basic protocol: no command byte, no state', async () => {
  const { b, log } = rig(FIB, { protocol: 'basic' }, { protocol: 'basic' });
  assert.deepEqual(await b.run([1, 1]), [8]);
  assert.deepEqual(log[0], ['tx', [1, 1]]);
  assert.throws(() => b.state(), BoardError);
  assert.throws(() => b.info(), BoardError);
});

test('traffic log carries both directions', async () => {
  const { b, log } = rig(FIB);
  await b.run([1, 1]);
  assert.deepEqual(log, [['tx', [CMD.RUN, 1, 1]], ['rx', [8]]]);
  assert.equal(hex(log[0][1]), '52 01 01');
});

test('a wrong lane count is rejected before anything is sent', () => {
  const { b, log } = rig(FIB);
  assert.throws(() => b.run([1]), BoardError);
  assert.equal(log.length, 0);
});

test('a silent board times out, with a TimeoutError carrying the byte count', async () => {
  const t = { write: async () => {}, read: async (n) => { throw new TimeoutError(n, 0); }, drain: async () => 0 };
  const b = new Board(t, FIB.manifest, { timeoutMs: 20 });
  await assert.rejects(b.run([1, 1]), e => e instanceof TimeoutError && e.got === 0);
});

test('a stalled request is dropped like the hardware does', async () => {
  const { t, b } = rig(FIB, { stallMs: 30 });
  await t.write([CMD.RUN, 1]);                    // command and one lane, then silence
  await new Promise(r => setTimeout(r, 80));
  assert.deepEqual(await b.run([1, 1]), [8]);     // framing recovered
});

test('bytes sent while the board is replying are ignored', async () => {
  const slow = async lanes => {            // keeps the board busy for a while
    await new Promise(r => setTimeout(r, 40));
    return evaluator(FIB)(lanes);
  };
  const t = new SimTransport(FIB.manifest, slow);
  const b = new Board(t, FIB.manifest, { timeoutMs: 500 });
  const done = b.run([1, 1]);
  await new Promise(r => setTimeout(r, 15));       // request is in, board is busy
  await t.write([CMD.RUN, 9, 9]);                  // would need a golden case if parsed
  assert.deepEqual(await done, [8]);
  await new Promise(r => setTimeout(r, 20));
  assert.equal(t.q.bytes.length, 0);               // and produced no reply
});

test('a corrupted state byte shows up as a mismatch', async () => {
  const { b } = rig(FIB, { fault: 5 });           // lane 5 is F5's byte
  const c = FIB.cases[0];
  await b.run(c.lanes);
  const nodes = stateToNodes(FIB.manifest, await b.state());
  const bad = Object.keys(nodes).filter(n => !sameValue(nodes[n], c.values[n]));
  assert.deepEqual(bad, [FIB.manifest.lanes.state[5].node]);
});

test('operations are serialised', async () => {
  const { b } = rig(FIB);
  const [r1, r2, r3] = await Promise.all([b.run([1, 1]), b.run([3, 5]), b.run([0, 9])]);
  assert.deepEqual([r1, r2, r3], [FIB.cases[0].reply, FIB.cases[2].reply, FIB.cases[3].reply]);
});

test('sweepVectors enumerates small spaces and samples large ones deterministically', () => {
  const all = sweepVectors(3, 2);
  assert.equal(all.length, 9);
  assert.equal(new Set(all.map(String)).size, 9);
  const a = sweepVectors(17, 4, { max: 50 });
  assert.equal(a.length, 50);
  assert.deepEqual(a, sweepVectors(17, 4, { max: 50 }));
  assert.deepEqual(a[0], [0, 0, 0, 0]);
  assert.deepEqual(a[1], [16, 16, 16, 16]);
  assert.ok(a.every(v => v.every(x => x >= 0 && x < 17)));
});

"""Multilane cmtz-core: reference model and scheduler (RFC-003, phase A0).

N lanes run independent instruction streams on one clock. Each lane is a
`sim.Machine` (the single-core reference, unchanged) whose field registers
are a view onto one shared physical register store, so a lease lends the
physical registers themselves, not copies:

  physical store   lanes x 32 field registers; lane l's register r is
                   P[32 l + r]. `LaneConfig.private` says how many of R1..
                   R31 physically exist in a lane (a register budget);
                   touching one beyond it is a fault (E_BUDGET).
  lease            `lease.Lease`: up to 8 of the owner's registers, seen by
                   the borrower as R24.. (its remote window). One lease per
                   borrower, one borrower per register, no nesting, and a
                   lane never lends and borrows at once (so the wait-for
                   graph has no cycle and nothing deadlocks).
  conflicts        before a lane executes an instruction it decodes the
                   field registers the instruction names (sources and
                   destination); if one is leased to another lane, or
                   poisoned, the lane stalls in EXEC until it is released.
  release          when the borrower's job HALTs, effective the next cycle.
                   A borrower that ends any other way (error, abort, step
                   limit, stuck on WAITI) poisons the lease instead: the
                   registers are never handed back as if unchanged, and an
                   owner that touches one is aborted (status error, code 7,
                   as the core reports a host abort). `clear_poison` is the
                   host's way back, with values it restored itself.

Timing follows the core's FSM (`backends/isa_core.py`, `Machine.cycles`):
FETCH and EXEC per instruction, MEM for a load, one more FETCH for a taken
trigger. A stall adds whole cycles in EXEC. Every cross-lane effect (a
lease taken, a lease released) becomes visible at a cycle boundary, so the
order lanes are processed in within a cycle does not change the result.
Without leases each lane's cycle count equals `Machine.cycles` for the same
run on its own; the tests check that.

The scheduler dispatches queued jobs to idle lanes under a policy:

  private   only variants that use no lease and fit the lane's budget
  static    a job's variant carries an exact lease (owner, registers),
            planned beforehand; the job waits until it can be acquired
  dynamic   (simulated only) the scheduler picks an owner and registers
            from each lane's `lendable` set at dispatch time, or falls back
            to a private variant, or waits

Acquisition refuses a lease whose registers hold a value outside [0, p)
(the restoration proofs say nothing about such values; see
`analysis/isa_clean.py`). With `snapshot=True` the leased values are
copied at acquire and compared at release; that copy is debug storage and
is reported separately (`Report.snapshot_bits`), never as workspace.

Randomised scheduling (`randomize=True`) draws job order, lane order and
start jitter from `seed`; every dispatch is logged in `decisions`, and
`replay=decisions` reproduces the run exactly without the generator.
The trace is a list of plain dicts; `analysis/lease_invariants.py` checks
it without importing this module.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..analysis.isa_catalytic import referenced_registers
from .encoding import NUM_R, NUM_X
from .image import CoreConfig, ProgramImage
from .lease import WINDOW, Lease, referenced, window_conflicts
from .sim import E_PC, E_STEPS, ERROR, HALTED, RUNNING, Machine

E_ABORT = 7             # the core's code for a host abort
E_BUDGET = 8            # simulator only: register beyond the lane's budget

FETCH, EXEC = "fetch", "exec"


class RegisterFault(Exception):
    pass


@dataclass
class LaneConfig:
    private: int = NUM_R - 1                  # R1..R{private} exist
    lendable: tuple[int, ...] = ()            # what `dynamic` may lend


@dataclass
class Variant:
    """One way to run a job. `lease` is an exact lease (static); `window`
    asks the dynamic policy for that many window registers."""
    image: ProgramImage
    kind: str = "private"
    lease: Optional[Lease] = None
    window: int = 0
    certificate: object = None                # analysis/lease_proof.py

    @property
    def borrows(self) -> int:
        return len(self.lease.regs) if self.lease else self.window


@dataclass
class Job:
    name: str
    variants: list[Variant]
    inputs: dict[str, int] = field(default_factory=dict)
    irq_at: tuple[int, ...] = ()
    arrival: int = 0
    lane: Optional[int] = None                # pinned to a lane
    after: tuple[str, ...] = ()               # jobs that must end first
    max_steps: int = 10_000_000
    expect: dict[str, int] = field(default_factory=dict)  # for the checks


@dataclass
class JobResult:
    name: str
    lane: int
    kind: str
    start: int
    end: int
    retired: int
    stalls: int
    status: int
    error: int
    outputs: dict[str, int]
    lease: Optional[Lease] = None
    restored: Optional[bool] = None           # snapshot compare at release
    model_cycles: int = 0                     # Machine.cycles (no stalls)

    @property
    def cycles(self) -> int:
        return self.end - self.start


@dataclass
class _Active:
    borrower: int
    lease: Lease
    phys: tuple[int, ...]
    acquired: int
    snapshot: Optional[list[int]] = None


class _RegView:
    """What a lane's Machine sees as R: logical registers mapped onto the
    physical store through the lane's lease and budget."""

    def __init__(self, mc: MultiCore, lane: int):
        self.mc, self.lane = mc, lane

    def __getitem__(self, r: int) -> int:
        i = self.mc.phys(self.lane, r)
        return 0 if i is None else self.mc.P[i]

    def __setitem__(self, r: int, v: int) -> None:
        i = self.mc.phys(self.lane, r)
        if i is not None:
            self.mc.P[i] = v

    def __len__(self) -> int:
        return NUM_R

    def __iter__(self):
        return (self[r] for r in range(NUM_R))


@dataclass
class _Lane:
    idx: int
    cfg: LaneConfig
    X: list[int]
    D: list[int]
    job: Optional[Job] = None
    variant: Optional[Variant] = None
    m: Optional[Machine] = None
    phase: str = FETCH
    t: Optional[int] = None                   # next action; None = blocked/idle
    start: int = 0
    stalls: int = 0
    stall_since: int = 0
    wants: frozenset = frozenset()            # physical regs it waits for
    irq: set = field(default_factory=set)
    busy: int = 0
    retired: int = 0
    stall_total: int = 0


@dataclass
class Report:
    results: list[JobResult]
    makespan: int
    lanes: list[dict]
    leases: dict
    snapshot_bits: int
    storage_bits: dict
    trace_hash: str
    pending: list[str] = field(default_factory=list)  # never dispatched

    @property
    def retired(self) -> int:
        return sum(r.retired for r in self.results)

    @property
    def ipc(self) -> float:
        return self.retired / self.makespan if self.makespan else 0.0

    def by_name(self) -> dict[str, JobResult]:
        return {r.name: r for r in self.results}


Policy = Callable[["MultiCore", int, int, list[Job]],
                  Optional[tuple[Job, int, Optional[Lease]]]]


class MultiCore:
    def __init__(self, core: CoreConfig, lanes: int = 2,
                 configs: Optional[list[LaneConfig]] = None,
                 policy: str | Policy = "private", seed: int = 0,
                 randomize: bool = False, jitter: int = 0,
                 dispatch_cycles: int = 0, snapshot: bool = False,
                 trace_access: bool = False,
                 replay: Optional[list[dict]] = None,
                 accept: Optional[Callable[[Variant, Lease], list[str]]] = None):
        self.core = core
        self.n = lanes
        cfgs = configs or [LaneConfig() for _ in range(lanes)]
        if len(cfgs) != lanes:
            raise ValueError("one LaneConfig per lane")
        self.P = [0] * (lanes * NUM_R)
        self.lanes = [_Lane(i, c, [0] * NUM_X, [0] * core.dmem_size)
                      for i, c in enumerate(cfgs)]
        self.holder: dict[int, int] = {}      # physical reg -> borrower lane
        self.poisoned: set[int] = set()
        self.active: dict[int, _Active] = {}  # borrower lane -> lease
        self.poison_of: dict[int, _Active] = {}   # borrower -> poisoned lease
        self.queue: list[Job] = []
        self.done: dict[str, JobResult] = {}
        self.trace: list[dict] = []
        self.decisions: list[dict] = []
        self.rng = random.Random(seed)
        self.randomize, self.jitter = randomize, jitter
        self.dispatch_cycles = dispatch_cycles
        self.snapshot, self.trace_access = snapshot, trace_access
        self._replay = list(replay) if replay is not None else None
        self.policy: Policy = (POLICIES[policy] if isinstance(policy, str)
                               else policy)
        self.accept = accept      # lease acceptance check (A3), or None
        self._releases: list[tuple[int, int, bool]] = []  # (t, lane, clean)
        self._dispatch_at: set[int] = set()
        self.stats = {"acquire": 0, "release": 0, "refuse": 0, "poison": 0,
                      "stall_events": 0, "peak_borrowed": 0,
                      "restored": 0, "not_restored": 0}
        self.t = 0

    # ── state ────────────────────────────────────────────────────────

    def scramble(self, seed: int, below: Optional[int] = None) -> None:
        """Random leftovers everywhere (R below `below`, default q)."""
        rng = random.Random(seed)
        hi = below or self.core.q
        for i in range(len(self.P)):
            self.P[i] = 0 if i % NUM_R == 0 else rng.randrange(hi)
        for ln in self.lanes:
            ln.X[:] = [0] + [rng.randrange(1 << 16) for _ in range(NUM_X - 1)]
            ln.D[:] = [rng.randrange(self.core.q) for _ in ln.D]

    def regs(self, lane: int) -> list[int]:
        """The lane's own physical registers (no window)."""
        return self.P[lane * NUM_R:(lane + 1) * NUM_R]

    def write_reg(self, lane: int, r: int, v: int) -> None:
        """Host write into a lane's own register (refused if leased)."""
        i = lane * NUM_R + r
        if i in self.holder or i in self.poisoned:
            raise RegisterFault(f"R{r} of lane {lane} is leased or poisoned")
        if not 0 <= v < self.core.q:
            raise ValueError(f"R{r} <- {v}: outside [0, q)")
        if r:
            self.P[i] = v

    def phys(self, lane: int, r: int) -> Optional[int]:
        if r == 0:
            return None
        a = self.active.get(lane)
        if a and WINDOW <= r < WINDOW + len(a.phys):
            return a.phys[r - WINDOW]
        if r > self.lanes[lane].cfg.private:
            raise RegisterFault(f"lane {lane}: R{r} is beyond its "
                                f"{self.lanes[lane].cfg.private} registers")
        return lane * NUM_R + r

    def _ev(self, t: int, ev: str, lane: int, **kw) -> None:
        self.trace.append({"t": t, "ev": ev, "lane": lane, **kw})

    # ── jobs and leases ──────────────────────────────────────────────

    def submit(self, *jobs: Job) -> None:
        names = {j.name for j in self.queue} | set(self.done)
        for j in jobs:
            if j.name in names:
                raise ValueError(f"job name {j.name!r} used twice")
            names.add(j.name)
            self.queue.append(j)
            self._dispatch_at.add(j.arrival)

    def fits(self, lane: int, v: Variant) -> bool:
        """The variant's private registers exist on the lane."""
        win = set(range(WINDOW, WINDOW + v.borrows))
        own = (referenced(v.image) | set(v.image.inputs.values())
               | set(v.image.outputs.values())) - win
        return max(own, default=0) <= self.lanes[lane].cfg.private

    def inputs_free(self, lane: int, v: Variant) -> bool:
        """The host can write the variant's inputs now: none of them is a
        register of this lane that is leased out or poisoned (the core's W
        refuses those)."""
        for r in v.image.inputs.values():
            if v.borrows and WINDOW <= r < WINDOW + v.borrows:
                return False
            i = lane * NUM_R + r
            if i in self.holder or i in self.poisoned:
                return False
        return True

    def lease_problems(self, borrower: int, lease: Lease) -> list[str]:
        """Why `lease` cannot be acquired for `borrower` now ([] if it can)."""
        errs = []
        o = lease.owner
        if not 0 <= o < self.n or o == borrower:
            return [f"owner lane {o} invalid for borrower {borrower}"]
        if borrower in self.active or borrower in self.poison_of:
            errs.append(f"lane {borrower} already holds a lease")
        if o in self.active or o in self.poison_of:
            errs.append(f"owner lane {o} is borrowing (a lane never lends "
                        "and borrows at once)")
        if any(i // NUM_R == borrower for i in set(self.holder) | self.poisoned):
            errs.append(f"lane {borrower} is lending")
        wanted = set().union(*(ln.wants for ln in self.lanes))
        for r in lease.regs:
            if r > self.lanes[o].cfg.private:
                errs.append(f"R{r} does not exist on lane {o}")
                continue
            i = o * NUM_R + r
            if i in self.holder:
                errs.append(f"R{r} of lane {o} is already leased")
            if i in self.poisoned:
                errs.append(f"R{r} of lane {o} is poisoned")
            if i in wanted:
                errs.append(f"R{r} of lane {o} has a lane waiting for it")
        return errs

    def acquire(self, borrower: int, lease: Lease, t: int) -> list[str]:
        errs = self.lease_problems(borrower, lease)
        if not errs:
            phys = tuple(lease.owner * NUM_R + r for r in lease.regs)
            vals = [self.P[i] for i in phys]
            if any(v >= self.core.p for v in vals):
                errs = [f"values {vals} outside [0, p)"]
        if errs:
            self.stats["refuse"] += 1
            self._ev(t, "refuse", borrower, owner=lease.owner,
                     regs=list(lease.regs), why=errs)
            return errs
        for i in phys:
            self.holder[i] = borrower
        self.active[borrower] = _Active(borrower, lease, phys, t,
                                        vals if self.snapshot else None)
        self.stats["acquire"] += 1
        self.stats["peak_borrowed"] = max(self.stats["peak_borrowed"],
                                          len(self.holder))
        ev = {"owner": lease.owner, "regs": list(lease.regs),
              "phys": list(phys)}
        if self.snapshot:
            ev["values"] = vals
        self._ev(t, "acquire", borrower, **ev)
        return []

    def _release(self, borrower: int, t: int) -> Optional[bool]:
        a = self.active.pop(borrower)
        for i in a.phys:
            del self.holder[i]
        vals = [self.P[i] for i in a.phys]
        ok = None
        if a.snapshot is not None:
            ok = vals == a.snapshot
            self.stats["restored" if ok else "not_restored"] += 1
        self.stats["release"] += 1
        ev = {"owner": a.lease.owner, "phys": list(a.phys)}
        if self.snapshot:
            ev.update(values=vals, restored=ok)
        self._ev(t, "release", borrower, **ev)
        self._wake(set(a.phys), t)
        return ok

    def _poison(self, borrower: int, t: int) -> None:
        a = self.active.pop(borrower)
        self.poison_of[borrower] = a
        for i in a.phys:
            del self.holder[i]
            self.poisoned.add(i)
        self.stats["poison"] += 1
        self._ev(t, "poison", borrower, owner=a.lease.owner,
                 phys=list(a.phys))

    def clear_poison(self, owner: int, values: dict[int, int]) -> None:
        """Host recovery: put restored values back into the owner's
        poisoned registers ({register: value}) and unlock them."""
        for r, v in values.items():
            i = owner * NUM_R + r
            if i not in self.poisoned:
                raise ValueError(f"R{r} of lane {owner} is not poisoned")
            if not 0 <= v < self.core.p:
                raise ValueError(f"R{r} <- {v}: outside [0, p)")
            self.P[i] = v
            self.poisoned.discard(i)
        for b, a in list(self.poison_of.items()):
            if not set(a.phys) & self.poisoned:
                del self.poison_of[b]
        self._ev(self.t, "clear_poison", owner, regs=sorted(values))

    def cancel(self, borrower: int) -> None:
        """Host: drop a lease whose borrower has not run since it was
        acquired (the registers come back untouched)."""
        a = self.active.pop(borrower)
        for i in a.phys:
            del self.holder[i]
        self._ev(self.t, "cancel", borrower, owner=a.lease.owner,
                 phys=list(a.phys))

    def _wake(self, phys: set[int], t: int) -> None:
        for ln in self.lanes:
            if ln.job and ln.t is None and ln.wants & phys:
                ln.stalls += t - ln.stall_since
                ln.t = t
                self._ev(t, "resume", ln.idx, stalled=t - ln.stall_since)

    # ── dispatch ─────────────────────────────────────────────────────

    def _eligible(self, lane: int, t: int) -> list[Job]:
        return [j for j in self.queue
                if j.arrival <= t and (j.lane is None or j.lane == lane)
                and all(a in self.done for a in j.after)]

    def _start(self, ln: _Lane, job: Job, vi: int, lease: Optional[Lease],
               t: int, start: Optional[int] = None) -> None:
        v = job.variants[vi]
        if start is None:
            start = t + self.dispatch_cycles + (
                self.rng.randrange(self.jitter + 1) if self.jitter else 0)
        self.decisions.append({"t": t, "lane": ln.idx, "job": job.name,
                               "variant": vi, "start": start,
                               "lease": None if lease is None else
                               [lease.owner, list(lease.regs)]})
        self.queue.remove(job)
        ln.job, ln.variant = job, v
        m = Machine(self.core, v.image)
        m.R, m.X, m.D = _RegView(self, ln.idx), ln.X, ln.D
        m.start()
        ins = {k: val for k, val in job.inputs.items() if k in v.image.inputs}
        missing = set(v.image.inputs) - set(ins)
        if missing:
            raise ValueError(f"{job.name}: no value for inputs {sorted(missing)}")
        if not self.inputs_free(ln.idx, v):
            raise RuntimeError(f"{job.name}: an input register is leased out")
        ln.m, ln.phase, ln.t, ln.start = m, FETCH, start, start
        ln.stalls, ln.wants, ln.irq = 0, frozenset(), set(job.irq_at)
        self._ev(t, "dispatch", ln.idx, job=job.name, kind=v.kind,
                 start=start, lease=None if lease is None
                 else [lease.owner, list(lease.regs)])
        try:
            m.set_inputs(ins)
        except RegisterFault as e:      # the host writes a missing register
            self._ev(t, "fault", ln.idx, why=str(e))
            self._fail(ln, start - 1, E_BUDGET)

    def _dispatch(self, t: int) -> None:
        idle = [ln for ln in self.lanes if ln.job is None]
        if self.randomize and self._replay is None:
            self.rng.shuffle(idle)
        for ln in idle:
            if self._replay is not None:
                d = self._replay[0] if self._replay else None
                if not d or d["t"] != t or d["lane"] != ln.idx:
                    continue
                self._replay.pop(0)
                job = next(j for j in self.queue if j.name == d["job"])
                lease = (None if d["lease"] is None
                         else Lease(d["lease"][0], tuple(d["lease"][1])))
                if lease and self.acquire(ln.idx, lease, t):
                    raise RuntimeError(f"replay: lease for {job.name} refused")
                self._start(ln, job, d["variant"], lease, t, d["start"])
                continue
            jobs = self._eligible(ln.idx, t)
            if not jobs:
                continue
            if self.randomize:
                self.rng.shuffle(jobs)
            pick = self.policy(self, ln.idx, t, jobs)
            if pick is None:
                continue
            job, vi, lease = pick
            self._start(ln, job, vi, lease, t)

    def try_lease(self, lane: int, v: Variant, lease: Lease, t: int) -> bool:
        """Acquire `lease` for running `v` on `lane`, after the window and
        acceptance checks. Used by the policies."""
        errs = window_conflicts(v.image, len(lease.regs))
        if not errs and self.accept is not None:
            errs = self.accept(v, lease)
        if errs:
            self.stats["refuse"] += 1
            self._ev(t, "refuse", lane, owner=lease.owner,
                     regs=list(lease.regs), why=errs)
            return False
        return not self.acquire(lane, lease, t)

    # ── execution ────────────────────────────────────────────────────

    def _finish(self, ln: _Lane, t: int) -> None:
        m, job = ln.m, ln.job
        end = t + 1
        lease = self.active[ln.idx].lease if ln.idx in self.active else None
        outs = {}
        for k, r in m.image.outputs.items():
            try:
                outs[k] = m.R[r]
            except RegisterFault:
                pass
        res = JobResult(job.name, ln.idx, ln.variant.kind, ln.start, end,
                        m.retired, ln.stalls, m.status, m.error, outs, lease,
                        None, m.cycles)
        self.done[job.name] = res
        ln.busy += end - ln.start
        ln.retired += m.retired
        ln.stall_total += ln.stalls
        self._ev(t, "end", ln.idx, job=job.name, status=m.status,
                 error=m.error, retired=m.retired, cycles=end - ln.start,
                 stalls=ln.stalls)
        if lease is not None:
            # a lease comes back only from a job that halted; anything else
            # poisons it. Both take effect at the next cycle boundary.
            self._releases.append((end, ln.idx, m.status == HALTED))
        ln.job = ln.variant = ln.m = None
        ln.t = None
        self._dispatch_at.add(end)

    def _fail(self, ln: _Lane, t: int, code: int) -> None:
        ln.m.status, ln.m.error = ERROR, code
        self._finish(ln, t)

    def _tick(self, ln: _Lane, t: int) -> None:
        m = ln.m
        if ln.phase == FETCH:
            if m.retired in ln.irq:
                ln.irq.discard(m.retired)
                m.pending = True
            if m.pending and not m.in_handler:
                m.pending = False
                m.cycles += 1
                if m.image.handler is not None:
                    m.saved_pc, m.pc = m.pc, m.image.handler
                    m.in_handler = True
                    m.dispatches.append((m.retired, m.saved_pc))
                    self._ev(t, "irq", ln.idx, retired=m.retired)
                ln.t = t + 1
                return
            if not 0 <= m.pc < len(m.code):
                m.cycles += 1
                return self._fail(ln, t, E_PC)
            ln.phase, ln.t = EXEC, t + 1
            return
        ins = m.code[m.pc]
        try:
            phys = [self.phys(ln.idx, r) for r in referenced_registers(ins)]
        except RegisterFault as e:
            self._ev(t, "fault", ln.idx, why=str(e))
            return self._fail(ln, t, E_BUDGET)
        blocked = frozenset(i for i in phys if i is not None and (
            i in self.poisoned or self.holder.get(i, ln.idx) != ln.idx))
        if blocked:
            ln.wants, ln.stall_since, ln.t = blocked, t, None
            self.stats["stall_events"] += 1
            self._ev(t, "stall", ln.idx, pc=m.pc, phys=sorted(blocked),
                     poisoned=bool(blocked & self.poisoned))
            return
        ln.wants = frozenset()
        if self.trace_access:
            self._ev(t, "access", ln.idx, phys=sorted(
                i for i in phys if i is not None))
        m.step()
        if m.status != RUNNING:
            return self._finish(ln, t)
        if m.retired >= ln.job.max_steps:
            return self._fail(ln, t, E_STEPS)
        if m.waiting and not m.pending and m.retired not in ln.irq:
            self._ev(t, "stuck", ln.idx, pc=m.pc)
            return self._fail(ln, t, E_ABORT)
        ln.phase = FETCH
        ln.t = t + (2 if ins.op == "LD" else 1)

    def run(self, max_cycles: int = 10 ** 9,
            stop_at: Optional[int] = None) -> Report:
        """Run until every lane is idle and nothing more can start (a later
        call continues from the same clock). `stop_at` pauses before that
        cycle, to look at the state mid-run."""
        t = self.t
        while True:
            if stop_at is not None and t >= stop_at:
                self.t = t
                return self.report()
            self.t = t
            for rel in sorted(r for r in self._releases if r[0] <= t):
                self._releases.remove(rel)
                _, b, clean = rel
                if not clean:
                    self._poison(b, t)
                    continue
                ok = self._release(b, t)
                if ok is not None:
                    for res in reversed(list(self.done.values())):
                        if res.lane == b and res.lease is not None:
                            res.restored = ok
                            break
            self._dispatch(t)
            for ln in self.lanes:
                if ln.job is not None and ln.t == t:
                    self._tick(ln, t)
            nxt = [ln.t for ln in self.lanes if ln.job is not None
                   and ln.t is not None]
            nxt += [r[0] for r in self._releases]
            if self._replay:
                nxt.append(self._replay[0]["t"])
            if self.queue:
                # a waiting job can only become startable when a lane frees
                # up, a lease ends or a job arrives
                nxt += [d for d in self._dispatch_at if d > t]
            self._dispatch_at = {d for d in self._dispatch_at if d > t}
            if not nxt:
                stuck = [ln for ln in self.lanes if ln.job is not None]
                if not stuck:
                    break
                # every running lane waits on a register nobody will free
                for ln in stuck:
                    if ln.wants & self.poisoned:
                        self._ev(t, "poison_abort", ln.idx, job=ln.job.name)
                        ln.stalls += t - ln.stall_since
                        self._fail(ln, t, E_ABORT)
                    else:
                        raise RuntimeError(f"deadlock: lane {ln.idx} waits "
                                           f"for {sorted(ln.wants)}")
                continue
            t = max(t + 1, min(nxt))
            if t > max_cycles:
                raise RuntimeError(f"no completion within {max_cycles} cycles")
        return self.report()

    # ── results ──────────────────────────────────────────────────────

    def storage(self) -> dict:
        """Physical storage in bits: private field registers (R0 is a
        constant, not storage), index registers, lease metadata per lane
        (valid, poisoned, owner, and a register number per window slot),
        and the debug snapshot (counted on its own)."""
        rw = self.core.rw
        owner_bits = max(1, (self.n - 1).bit_length())
        meta = self.n * (2 + owner_bits + 8 * 5 + 4)
        return {"field_registers": sum(ln.cfg.private for ln in self.lanes) * rw,
                "index_registers": self.n * (NUM_X - 1) * 16,
                "lease_metadata": meta,
                "poison_bits": self.n * NUM_R}

    def report(self) -> Report:
        res = list(self.done.values())
        makespan = (max(r.end for r in res) - min(r.start for r in res)
                    if res else 0)
        lanes = [{"lane": ln.idx, "busy": ln.busy, "retired": ln.retired,
                  "stalls": ln.stall_total, "private": ln.cfg.private}
                 for ln in self.lanes]
        h = hashlib.sha256(json.dumps(self.trace, sort_keys=True,
                                      default=str).encode()).hexdigest()
        snap = (self.stats["peak_borrowed"] * self.core.rw
                if self.snapshot else 0)
        return Report(res, makespan, lanes, dict(self.stats), snap,
                      self.storage(), h, [j.name for j in self.queue])


# ── policies ─────────────────────────────────────────────────────────────────

def private_only(mc: MultiCore, lane: int, t: int, jobs: list[Job]):
    for job in jobs:
        for vi, v in enumerate(job.variants):
            if not v.borrows and mc.fits(lane, v) and mc.inputs_free(lane, v):
                return job, vi, None
    return None


def static_leases(mc: MultiCore, lane: int, t: int, jobs: list[Job]):
    """A job runs its first variant that fits: a planned lease if it can be
    acquired now, or a private variant; otherwise it waits."""
    for job in jobs:
        for vi, v in enumerate(job.variants):
            if not mc.fits(lane, v) or not mc.inputs_free(lane, v):
                continue
            if v.lease is None and not v.window:
                return job, vi, None
            if v.lease is not None and not mc.lease_problems(lane, v.lease):
                if mc.try_lease(lane, v, v.lease, t):
                    return job, vi, v.lease
    return None


def find_lease(mc: MultiCore, lane: int, count: int) -> Optional[Lease]:
    """First owner (lane order) with `count` free lendable registers that
    hold field values."""
    for o in range(mc.n):
        if o == lane:
            continue
        free = []
        for r in mc.lanes[o].cfg.lendable:
            i = o * NUM_R + r
            if (i not in mc.holder and i not in mc.poisoned
                    and mc.P[i] < mc.core.p):
                free.append(r)
        if len(free) >= count:
            lease = Lease(o, tuple(free[:count]))
            if not mc.lease_problems(lane, lease):
                return lease
    return None


def dynamic_leases(mc: MultiCore, lane: int, t: int, jobs: list[Job]):
    """Prefer a variant that borrows (the lease is chosen now from the
    lendable registers), else a private one that fits, else wait."""
    for job in jobs:
        for vi, v in enumerate(job.variants):
            if v.window and mc.fits(lane, v) and mc.inputs_free(lane, v):
                lease = find_lease(mc, lane, v.window)
                if lease and mc.try_lease(lane, v, lease, t):
                    return job, vi, lease
        for vi, v in enumerate(job.variants):
            if not v.borrows and mc.fits(lane, v) and mc.inputs_free(lane, v):
                return job, vi, None
    return None


POLICIES: dict[str, Policy] = {"private": private_only,
                               "static": static_leases,
                               "dynamic": dynamic_leases}

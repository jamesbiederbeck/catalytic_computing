# RFC-003: Catalytic Multicore Scheduler and Multilane Processor

**Status:** Implemented through A5 (A4 in the simulator only); see *Results by phase* and *Conclusions*  
**Affects:** `cmtz/isa/` (new multilane model, leases), `cmtz/analysis/`
(lease invariants, lease proofs), `cmtz/backends/` (multilane RTL and UART),
`examples/fpga/` (board tools), the `tang-nano-20k` playground (new APIO
project)  
**Baseline:** cmtz-core as of `d56477a` (projection programs, clean-evaluation
proof, verifier fixes, cycle model, benchmarks). The single-core design, its
ISA, its UART protocol and its tests are unchanged.

---

## Two hypotheses, kept apart

1. **Lanes.** Several independent instruction streams on one chip complete
   more jobs per second than one stream. This is a throughput claim, and the
   only evidence that counts is retired instructions per clock, measured.
2. **Borrowing.** A lane that borrows another lane's *occupied* registers
   (values that must come back) can run with less private workspace, without
   adding physical storage. This is a storage claim. It holds only if the total
   physical storage stays fixed and the borrowed registers really come back.

A result for one says nothing about the other. Every table states which one it
measures.

## Model

**Lanes.** Each lane has its own pc, FSM state, field registers R0..R31,
index registers X0..X15, program memory and data memory. All lanes use the
cmtz-core ISA (`cmtz/isa/encoding.py`) and its semantics (`sim.Machine`). The
reference model composes one `Machine` per lane. Each lane's `R` is a view
onto a shared physical register store, so no ISA semantics are written twice.

**Physical register store.** It holds `lanes × 32` field registers. Lane *l*'s
register *r* is `P[32l + r]`. A lane may have a *private budget* (`LaneConfig.
private`): only R1..R*budget* physically exist, and touching one beyond that
is a fault. The budget is how the simulator models a smaller register file for
hypothesis 2.

**Lease.** A lease gives up to 8 of the owner lane's registers to a borrower
lane. The borrower reaches them through its *remote window*, R24..R24+n−1.
A lease names the owner's physical registers, never a copy. Restrictions:

- one outstanding lease per lane;
- one borrower per register;
- no nesting;
- no lane lends and borrows at the same time;
- no cross-lane data memory;
- a window register outside the lease, and a host input written into the
  window, are both refused.

**Why R24..R31.** The compiler never allocates there (`MAIN_POOL` is
R1..R23), and the runtime library (`lib/matrix.s`, matpow) owns that range.
A program that links the library cannot borrow, and `window_conflicts`
refuses it. Placing the window there respects the library's reservation
instead of moving it.

**Conflicts.** Before a lane executes an instruction, it decodes the field
registers the instruction names: its sources and its destination. It uses
`referenced_registers` from `encoding.py`'s operand formats. If one of them
is leased to another lane, or poisoned, the lane stalls in EXEC and retries
every cycle. The host cannot write a leased register either.

**Acquire and release.**

- Acquisition refuses a lease when any of the owner's registers holds a value
  outside [0, p). Every restoration proof assumes field values (see
  `analysis/isa_clean.py`; on this hardware 17 and 18 come back as 0 and 1).
- A lease is released when its borrower's job HALTs, effective at the next
  clock edge.
- A borrower that ends any other way (an error, an abort, the step limit, or
  stuck on WAITI) *poisons* the lease. The registers are never returned as if
  unchanged. An owner instruction that touches a poisoned register stalls,
  and the scheduler (in hardware, the host) aborts that job. `clear_poison` is
  the host's recovery path: it writes values the host restored itself, then
  unlocks the registers.

**Deadlock.** A lane that lends cannot borrow, and a lane that borrows cannot
lend. So an owner waits only on borrowers, and a borrower never waits on a
lease. The wait-for graph has no cycle. The forbidden case is A borrowing from
B while B borrows from A, with both then touching their own lent registers.
A test builds it and shows it is refused. Starvation: a register that a
stalled owner is waiting for cannot be leased again until that owner has run
the instruction.

**Cycles.** Timing follows the core's FSM:

- FETCH and EXEC for every instruction;
- MEM for a load;
- one extra FETCH for a taken trigger;
- whole cycles added in EXEC for a stall.

Every cross-lane effect (a lease taken, a lease released) becomes visible at
a cycle boundary, so the order in which lanes are processed within a cycle
does not change the result. Without leases, each lane's cycle count equals
`Machine.cycles` for the same run on its own.

## Phases

| Phase | Content | Where |
|---|---|---|
| A0 | Python multilane model and scheduler (policies: private, static, dynamic), independent invariant checker, seeded randomised scheduling and replay | `cmtz/isa/multicore.py`, `cmtz/isa/lease.py`, `cmtz/isa/workloads.py`, `cmtz/analysis/lease_invariants.py` |
| A1 | Two independent lanes in generated RTL; a new UART front end (protocol `CMTM`) that loads, starts and reads each lane; on-chip cycle stamps | `cmtz/backends/isa_mlane.py`, playground `projects/cmtz/cmtz-dual/` |
| A2 | Static leases in hardware: banked 8-register window, owner stall, range check at acquire, poison | the same generator with leases enabled |
| A3 | Proof-aware leases: a lease is accepted only with a restoration proof of the allocated machine code under the actual register mapping | `cmtz/analysis/lease_proof.py` |
| A4 | Dynamic leases from the program (BORROW/RELEASE), in the simulator | `cmtz/isa/multicore.py` |
| A5 | Benchmarks: throughput (hypothesis 1) and fixed-storage register budgets (hypothesis 2) | `examples/fpga/mlane_bench.py` |

## Results by phase

Every phase lists what it checked and how: by mathematics or proof, in the
ISA simulator, in RTL simulation, or on the physical FPGA.

### A0: reference model and scheduler

`cmtz/isa/multicore.py` holds the model, the scheduler and the three
policies. `cmtz/isa/lease.py` holds leases, relocation and window checks.
`cmtz/isa/workloads.py` builds the jobs, and
`cmtz/analysis/lease_invariants.py` is the independent checker.

- **Mathematics or proof.** None in this phase. Restoration is *observed*
  (snapshot compare), not proven; A3 adds the proof.
- **Simulator** (`test_multicore.py`, 35 tests):
  - On 1, 2, 4 and 8 lanes, each lane's final registers, index registers,
    data memory, status, retired count and cycle count equal a scalar
    `Machine` run of the same program from the same state.
  - Leased `gate_clean` (masks in another lane's registers, mapped at random)
    returns the owner's registers bit for bit. Its output equals the oracle.
  - The owner stalls exactly from the EXEC of its first instruction that
    names a leased register until the cycle after the borrower's HALT. This
    holds for a read and for a write.
  - Refusals: a value outside F_p, an image that links the matpow library,
    nine registers, a register named twice, R0, a lane that lends and
    borrows, and the A↔B cycle.
  - A borrower that faults after writing into the window poisons the lease.
    The owner that then touches the register is aborted, the registers
    cannot be leased again, and `clear_poison` recovers them.
  - 12 seeded random runs (2–4 lanes, dynamic policy, jitter, gate, tree and
    owner jobs): every invariant holds, every job's output equals the
    oracle, every lease comes back.
  - Replay from the recorded decisions, with no random generator, gives the
    same trace hash; a different seed does not.
  - The invariant checker catches six hand-corrupted traces: wrong values,
    an early release, a double lease, an owner touching a leased register,
    a missing release and a wrong stall sum.
- **RTL and board.** Not applicable.

### A1: two independent lanes

The new generator is `cmtz/backends/isa_mlane.py`, and the CMTM protocol is
in `cmtz/isa/mlane_host.py`. The lanes are *unmodified* `cmtz_core`
instances: the generated lane source is byte for byte the single core's, and
a test pins the sha256 of `cmtz_core.v` to what the single-core board
regression ran on. Each lane has its own program memory, data memory,
registers and control. A free-running counter stamps each lane's first
running cycle and the first cycle after its run; their difference is the
run's cycle count, measured on the chip.

- **Mathematics or proof.** None needed: the lanes share nothing but the
  clock.
- **Simulator.** For every independent scene, the multilane model's
  expectation equals a scalar `Machine` per lane (`test_isa_mlane.py`).
- **RTL** (iverilog):
  - 7 scenes on 2 lanes and on 4 lanes: different programs started together,
    staggered starts, interrupts at given retired counts on two lanes, two
    compiled examples, and a faulting lane beside a working one. Every lane's
    registers, index registers, changed data words, status, retired count,
    cycle stamps and stall count match the model.
  - The UART testbench passes the same scenes started with `B`, plus the
    protocol's refusals.
- **Board** (Tang Nano 20K, `cmtz-dual`, SRAM, `-noalu`, `mlane_run.py`):
  958 of 958 checks.
  - 25 ordered pairs of the five projection programs, each lane alone and
    both together. Every lane is compared with the model seeded from the
    board's state, with a scalar `Machine`, and with the oracle.
  - An interrupt is sent over the UART to lane 0 while it runs `gate_irq`
    (2.46 M instructions; T landed at 1.14 M and 0.97 M). Lane 0's result
    is right with the restoring handler and wrong without it. Lane 1's
    `tree_clean` is cycle-for-cycle unchanged (39 910 cycles both times).

**Throughput (hypothesis 1), measured on the chip.**

- One lane retires 0.500 instructions per clock: the FSM takes 2 cycles per
  instruction, and none of these programs loads. That is the ceiling per
  lane.
- Two lanes reach 1.000 only when the two jobs are equally long:
  `gate_clean`+`gate_clean`, `tree_clean`+`tree_clean`, and so on.
- With unequal jobs the gain is what the shorter job hides. `tree_clean`
  beside `gate_direct` gives 0.500, because the lane that finishes early
  idles.
- For a 12-job random batch dealt in pairs, the makespan went from 203 088
  cycles to 160 868, a 1.26× gain.
- Wall-clock time over the UART got *worse*, 0.14 s against 0.18 s. Every
  job here runs for at most 1.5 ms, and the host's polling dominates.

Two lanes double the peak rate, not the delivered rate. Delivering it needs
a scheduler that keeps both lanes busy, and jobs long enough that the UART
is not the bottleneck.

**Cost** (yosys + nextpnr, `-noalu`):

| Build | LUT4 | DFF | BSRAM | MULT | Fmax (MHz) |
|---|---|---|---|---|---|
| `cmtz-core` (1 lane) | 2918 | 690 | 2 | 2 | 114.9 |
| `cmtz-dual` (2 lanes) | 5030 | 1132 | 4 | 4 | 104.0 |

The board clock is 27 MHz, so every build has timing slack.

### A2: static register leases in hardware

`emit_mlane(..., lease=True)` turns each lane into `emit_core(...,
lane=True)`: the same datapath, with the field registers moved out into
one physical array `RF[32 l + r]`, a registered write port, and a `stall`
input that holds EXEC. Per borrower lane, the lease is state (none, active
or poisoned), owner, bank (1..3) and count (1..8). The borrower's window
R24..R24+count−1 *is* the owner's R(8·bank)..: the same flip-flops.

- The owner stalls when its instruction names a register held by an active
  or poisoned lease. The operand-use decode comes from `encoding.FORMATS`.
- `K` acquires a lease, with both lanes idle. It refuses (`FE`) a value
  outside F_17, and (`FF`) a lane that would lend and borrow, a bank
  already taken, or a second lease.
- A borrower that halts releases the lease in the cycle its status changes,
  so the owner runs again one cycle after the borrower's HALT, as in the
  model. A borrower that errors poisons the lease. `K b o 0 0` cancels an
  unused lease or clears a poisoned one; the host restores the values.
- Host writes to leased registers are refused.

The hardware leases whole banks, and the model leases arbitrary register
sets. `hw_lease_code` states the hardware rule. The testbench generator
asserts that the model accepts every lease the hardware accepts.

- **Mathematics or proof.** Not in this phase: restoration is observed
  bit for bit, not proven. A3 adds the proof.
- **Simulator** (`test_multicore.py`): the borrower works in place.
  Sampled mid-run, the owner's own physical registers hold the borrower's
  intermediate values (at least 6 distinct states over one `gate_clean`).
  The borrower's accesses go to the owner's register indices, never to its
  own R24..R26, and the values are back at the end.
- **RTL** (iverilog, `test_isa_mlane.py`), on 2 and on 4 lanes:
  - 14 scenes on 2 lanes and 15 on 4 (two leases at once): a gate in the
    owner's live registers while the owner runs and then stalls (369 cycles,
    as predicted); a gate tree in the other lane's bank 1 while that lane
    loops and then stalls (39 742 cycles); an owner write that stalls; every
    refusal code; a poisoned lease, then cleared and restored; a cancelled
    lease.
  - Every lane's full state, cycle count and stall count match the model.
  - Expecting one stall cycle more makes the testbench print FAIL.
  - The UART testbench passes the same scenes through `K`/`E`, plus a host
    write to a leased register (refused) and a cancel.
- **Board** (`cmtz-lease`, `mlane_run.py`): 1733 of 1733 checks.
  - The A1 checks all pass on this build too.
  - The lease scenes pass.
  - 40 random leases of `gate_clean` (3 registers) or `tree_clean` (5
    registers) into the other lane's bank 1 or 2, with random live values,
    while that lane runs a loop of random length and then reads a leased
    register. The owner's registers came back in 40 of 40, and the outputs
    equal the oracle.
  - Owner stalls ranged from 0 cycles (the owner reached the register
    after the borrower finished) to 39 425 cycles (it waited through a
    whole tree), each equal to the model's prediction.

**Cost.** Leases are expensive in logic:

| Build | LUT4 | DFF | BSRAM | Fmax (MHz) |
|---|---|---|---|---|
| `cmtz-dual` | 5030 | 1132 | 4 | 104.0 |
| `cmtz-lease` | 8822 | 1592 | 4 | 68.6 |

- That is +3792 LUT4 (+75 %) and +460 DFF.
- Fmax falls by a third, still well above the 27 MHz board clock.
- In `cmtz-core` the register file maps to distributed RAM (98 RAM16SDP4 per
  core). Shared across lanes, with a window mux in front of three read ports
  per lane, it becomes 320 flip-flops behind wide multiplexers.
- The register bits did not change: 2 × 32 × 5 in both builds.
- The lease metadata is 2 × (2 + 1 + 2 + 4) = 18 flip-flops, plus the
  per-lane cycle stamps.

So in hardware, borrowing saves no storage at this register-file size. It
*costs* logic. Any storage claim has to come from A5.

### A3: proof-aware leases

`cmtz/analysis/lease_proof.py`. A borrowed computation carries a *claim*:
the polynomial each output gains, and the private registers it may leave
changed. The checker never trusts it. `certify(image, window, claim)` runs
the clean-computation proof (`isa_clean.verify_clean`, polynomial symbolic
execution over F_17 with x^17 = x) on the borrower's machine code as
allocated. The window registers R24..R24+n−1 are the catalytic set whose
entry values the lease supplies (`verify_clean(..., supplied=...)`; nothing
else may be read unless the host writes it). The resulting `Certificate`
records:

- the program's crc32;
- the window size and the catalytic registers;
- the proven output gains and the permitted temporaries;
- the initial-value domain (`requires_canonical`, which always includes the
  window);
- the exact retired and cycle counts, which bound how long the owner can
  be stalled;
- the handler status.

`check_lease` binds a certificate to an actual lease:

- the same program (crc) and the same register count;
- owner ≠ borrower, so the window cannot alias the borrower's own
  registers;
- distinct registers;
- an owner whose interrupt handler names none of them.

The physical mapping is a renaming of logical R24..: distinct registers of
another lane, written by nobody else while the lease is held (exclusive
lease plus owner stall). The proof therefore carries over to every mapping
that passes these checks. `Checker` plugs into `MultiCore(accept=...)`,
proves each (program, window, claim) once, and records the certificate key
in the trace's `acquire` event.

Refused, each with a test (`test_lease_proof.py`, 19 tests):

| Case | Why |
|---|---|
| 15 rotations instead of 16 | a borrowed register ends as ω^15·τ |
| uncompute removed | masks come back, but y gains ĝ at 8a: wrong gain |
| rotation by −1 (order 2) | the τ1·τ2 term survives |
| wrong claimed gain; undeclared scratch register | output or scratch effect not as claimed |
| touches R27 with a 3-register window; host input into the window; window too small | ownership |
| a program that links matpow | the library owns R24..R31 |
| `LD`, `ST`, `BZ` on a window register | unsupported memory or data-dependent control: fail closed |
| reads R13, which neither host nor lease supplies | leftover, possibly in [p, q) |
| handler names a window register | interrupt effect on borrowed state |
| certificate for another program; wrong register count; owner = borrower; owner's handler names a leased register | binding |
| a variant without a claim | no lease on a program's word |

Accepted, also tested: rotation by ω² (order 8) or by 4 (order 4). Over
16 steps any root of order greater than 2 cancels the degree-2 cross terms.
A handler that passes `verify_handler` and stays off the window is accepted
too.

- **Mathematics or proof.** Restoration and the output gain are proven for
  every value in [0, p) of every register, on the exact machine code the
  lane runs.
- **Simulator.** In seeded dynamic runs with the checker, every acquire
  carries a certificate key, every job's output equals the oracle, and no
  lease comes back changed. A non-restoring variant is refused and the job
  runs on its private variant instead.
- **RTL.** No new hardware: acceptance is the host's or the scheduler's
  job. The range check it relies on is in the RTL (A2).
- **Board** (`cmtz-lease`, `mlane_run.py --only leases`): 626 of 626 checks.
  The host certified the leased `gate_clean` and `tree_clean` before
  sending `K`, bound each of 40 random leases to its certificate, and
  refused a 15-rotation mutant before sending anything. The standard lease
  scenes deliberately run uncertified programs, to drive the hardware's
  poison path.

A debug snapshot (`MultiCore(snapshot=True)`) compares leased values at
release. Its storage (`Report.snapshot_bits`) is reported separately and
never counted as workspace.

### A4: dynamic leases (simulator only)

**Opcodes.** `BORROW owner, bank, count` (opcode 28, three small
immediates in fields a, b, c) and `RELEASE` (opcode 29, no operands). They
sit in `encoding.EXT_OPS`, *outside* `OPS`:

- the single-core generator, assembler and simulator do not know them;
- `cmtz_core.v` is unchanged, and its hash test still passes;
- a single core that meets one reports a bad opcode, as the RTL would;
- `assemble(..., ext=True)`, `decode(..., ext=True)` and
  `Machine(..., ext=True)` accept them, and `multicore.py` executes them.

The format follows the A2 hardware lease: bank and count, with the owner as
a lane number.

**Semantics** (`MultiCore._lease_op`):

| Case | Behaviour |
|---|---|
| a lease that is free | acquired in EXEC; window R24.. maps to the owner's bank from the next instruction on |
| busy: registers held, poisoned, or wanted by an older waiter; owner borrowing; this lane lending | stall in EXEC and retry at every lease change |
| two requests in one cycle | lane order |
| waiters | served oldest first, so no request is overtaken forever (tested with three lanes) |
| invalid: owner = self or no such lane, bank ∉ 1..3, count ∉ 1..8, registers the owner lacks | fault, `E_LEASE` |
| second lease (double borrow, or a poisoned one still held) | fault |
| a leased value outside [0, p) | fault (fail closed) |
| `RELEASE` | the registers come back at the next cycle boundary; the lane runs on |
| `RELEASE` without a lease (double release) | fault |
| `HALT` while holding | release, as in A2 |
| any other end while holding (fault, abort, step limit) | poison |
| owner halts while lending | allowed; its next job stalls on the registers until they come back |

**Deadlock.** The A0 rule still holds: a lane never lends and borrows at
once. A lender that executes `BORROW` waits until its own registers come
back. Every wait therefore ends at a borrower, which never waits; the
three-lane case is tested.

**Proof.** `run_symbolic(..., ext=True)` gives the window fresh variables
at `BORROW`, setting the lane's own R24.. values aside. At `RELEASE`, or
at `HALT` while borrowing, each borrowed register must equal its variable
again, and then the private values come back.
`lease_proof.certify_dynamic` runs the clean-computation proof on that
basis. A `Checker` in the scheduler refuses to *dispatch* a program that
contains `BORROW` until it is certified. Refused cases:

- a register not restored at `RELEASE`, or at `HALT`;
- a nested `BORROW`, or `RELEASE` without a lease;
- a branch on a borrowed value, or `ST` of one;
- a handler in a program that borrows.

**Policy** `least_cost` (`MultiCore(policy="cost")`) chooses between
borrowing, waiting, spilling, recomputing or running privately. For each
variant it estimates the finish time:

- start now for anything that fits the lane's budget, or for a lease that
  is free now;
- otherwise, for a busy lease, the expected release plus the run time.

If waiting is quickest, the job stays queued. On a tie it does not borrow.
In the tests:

- with a 7-register budget it picks the gate's spill form (19 cycles) over
  borrowing (614);
- with only borrowing able to fit, it borrows;
- with a 10-register budget it runs the clean gate privately rather than
  borrow;
- two trees on 17-register lanes, with one bank to lend: the second waits
  for the lease.

- **Mathematics or proof.** Dynamic programs are proven restoring, between
  `BORROW` and `RELEASE`, for every register value in [0, p).
- **Simulator** (`test_dynamic_leases.py`, 33 tests): every case in the
  table above, the policy cases, and 4 seeded random mixes on 4 lanes
  (16 dynamic jobs each, random owners, arrivals and tail lengths). Every
  lease came back, every output matched the oracle, and no invariant broke.
- **RTL and board.** Not built. The A2 hardware already takes 75 % more
  logic than independent lanes. A runtime `BORROW` adds a request queue and
  arbitration in front of the same array, and nothing measured so far
  (A5) makes leases pay for themselves. This is the open blocker, not a
  claim.

### A5: benchmarks

`examples/fpga/mlane_bench.py` reproduces every model number here (seed 7,
about 8 minutes). `test_storage_benchmark_gate` pins the gate storage
result. Board numbers come from `mlane_run.py --json` runs on the chip.

**Hypothesis 1: lanes.** The model was checked cycle-for-cycle against the
RTL and the board. 40 queued jobs, a random mix of the five projection
programs:

| Lanes | Makespan (cycles) | IPC | Jobs/s @27 MHz | Mean latency | p95 latency |
|---|---|---|---|---|---|
| 1 | 367 674 | 0.500 | 2 937 | 7.94 ms | 13.6 ms |
| 2 | 202 390 | 0.908 | 5 336 | 3.88 ms | 6.12 ms |
| 4 | 120 612 | 1.524 | 8 954 | 1.82 ms | 3.04 ms |
| 8 | 79 820 | 2.303 | 13 530 | 0.62 ms | 1.53 ms |

- Scaling falls short of linear, because one `tree_clean` (39 910 cycles)
  bounds the makespan once there are enough lanes.
- With the measured host cost of about 6 ms per job (UART load, inputs,
  start, poll), one lane manages 162 jobs/s and eight manage 1 482. Lanes
  then mainly hide host latency, which a host that pipelines its UART
  traffic across lanes would also do.

On the board, with lockstep groups (each group of n jobs started with `B`
and awaited before the next):

| Build | Batch | One lane | All lanes | Gain |
|---|---|---|---|---|
| `cmtz-dual` | 12 jobs | 203 088 cycles | 160 868 | 1.26× |
| `cmtz-lease` | 12 jobs | 242 194 | 199 974 | 1.21× |
| `cmtz-quad` | 12 jobs | 85 010 | 80 434 | 1.06× |

The lockstep groups waste most of the extra lanes. The queue scheduling
above is what lanes are worth, and on the board it needs a host that keeps
every lane fed. Two equal jobs reach exactly 2× (IPC 1.000 measured).
Wall time was UART-bound in every board run: lanes did not make the board
batch faster.

**Hypothesis 2: borrowing at fixed physical storage** (model, both
streams). Lane 0 holds five live values in R16..R20 while it loops, and
reads them at the end. Lane 1 runs a stream of gates or of gate trees.
The table gives the smallest lane-1 register budget each policy can run the
stream with:

| Stream | Policy | Lane-1 registers | Total register bits | Spill bits | Lease metadata bits | Borrowed | Cycles per job |
|---|---|---|---|---|---|---|---|
| gate | private, direct | 8 | 140 | 0 | 0 | 0 | 14 |
| gate | private, clean | 10 | 150 | 0 | 0 | 0 | 614 |
| gate | private, spill | 7 | 135 | 5 | 0 | 0 | 19 |
| gate | lease (static) | 7 | 135 | 0 | 94 | 3 | 614 |
| gate | lease (dynamic policy: the scheduler picks the owner's registers) | 7 | 135 | 0 | 94 | 3 | 614 |
| gate | BORROW/RELEASE in the program (A4) | 7 | 135 | 0 | 94 | 3 | 618 |
| gate | least cost | 7 | 135 | 5 | 0 | 0 | 19 (picked spill) |
| tree | private, direct | 20 | 200 | 0 | 0 | 0 | 34 |
| tree | private, clean | 22 | 210 | 0 | 0 | 0 | 39 910 |
| tree | private, spill | 17 | 185 | 25 | 0 | 0 | 53 |
| tree | lease (static) | 17 | 185 | 0 | 94 | 5 | 39 910 |
| tree | least cost | 17 | 185 | 25 | 0 | 0 | 53 (picked spill) |

Notes on the table:

- **Lease metadata.** The 94 bits are the model's register-list lease (8
  register numbers per lane). The banked hardware lease of A2 is 18
  flip-flops for two lanes.
- **Dynamic rows.** Both dynamic forms reach the static lease's budget and
  speed. `BORROW` and `RELEASE` add 4 cycles per job.
- **Tree stream.** No `BORROW`/`RELEASE` tree program was written, so the
  tree stream has no such row.

Findings:

- Borrowing does what it promises. It runs the computation with lane 1
  below the direct form's register count: 7 instead of 8 for a gate, 17
  instead of 20 for a tree. That is 5 and 15 register bits less, with
  every borrowed register proven and observed to come back. The owner did
  not stall here, because its loop outlasted each lease.
- It is the wrong way to get that saving:
  - **Speed.** Spilling the same temporaries to data memory reaches the
    same register count at 19 cycles per gate instead of 614 (32×), and
    53 per tree instead of 39 910 (750×).
  - **Storage.** Spilling uses 5 or 25 bits of data memory, which the
    lane has anyway (1024 words). Leases need 94 bits of metadata in the
    model's register-list form.
  - **Logic.** In hardware, leases cost 3792 LUT4 and 460 DFF on two lanes
    (A2). That is three orders of magnitude more than the 5 to 15 bits
    they save.
- The least-cost policy, offered every form, never chose to borrow.
- The clean (projection) forms are the slow ones whether borrowed or not,
  as in the projection write-up. Cook–Mertz pays in time for space, and at
  depth ≤ 2 the space it saves is a handful of temporaries.
- Snapshot storage (the debug compare at release, 15 or 25 bits) is
  reported separately and in none of these totals.

**Hardware** (`-noalu`, GW2AR-18: 20 736 LUT4):

| Build | LUT4 | DFF | BSRAM | MULT (18×18 + 9×9) | Fmax (MHz) | Board |
|---|---|---|---|---|---|---|
| `cmtz-core` | 2 918 | 690 | 2 | 1 + 1 | 114.9 | 2459/2459 (rerun at the end of this work) |
| `cmtz-dual` | 5 030 | 1 132 | 4 | 2 + 2 | 104.0 | 958/958 |
| `cmtz-lease` | 8 822 | 1 592 | 4 | 2 + 2 | 68.6 | 1733/1733, plus 626/626 proof-gated |
| `cmtz-quad` | 9 048 | 1 627 | 8 | 4 + 4 | 110.9 | 468/468 (quick) + 252/252 |
| `cmtz-quad-lease` | — | — | — | — | — | does not place: "Unable to find legal placement for all cells, design is probably at utilisation limit" |

Independent lanes cost close to one core each: 2 918, then about 2 060 more
per lane. Leasing on four lanes does not fit the chip at all.

## Conclusions

1. **Lanes work and are cheap.** Two to four independent lanes are correct
   on the chip, cost about 2 000 LUT4 each, and raise peak IPC from 0.5 to
   0.5 × lanes. Delivered throughput depends on keeping the lanes fed. With
   a job queue, the model gives 1.82× on 2 lanes, 3.05× on 4 (IPC 1.52)
   and 4.6× on 8. The board's lockstep batches reached 1.06–1.26×. Over
   the UART the host is the bottleneck.
2. **Borrowing is correct, and a net loss here.**
   - It works on the chip: restoration is proven before a lease and
     observed after every one, values outside F_17 are refused, a failure
     poisons, and owner stalls match the model to the cycle.
   - It saves the same registers that spilling to existing data memory
     saves, 32× to 750× more slowly.
   - Its hardware costs 75 % more logic on two lanes, and does not fit on
     four.
   - No controlled experiment here shows a memory improvement, so none is
     claimed.

## Open problems and next blockers

- **When could borrowing pay?** Only where the registers saved are worth
  more than spill bandwidth and lease logic: deep trees, where Cook–Mertz
  reuses one set of catalytic registers across levels (depth 3 needs
  coefficients out of registers first; see the projection write-up), or a
  machine without data memory.
- **Cheaper leases.** A bank-granular lease whose window is muxed only at
  the write-back and operand stage might keep the register file in
  distributed RAM. That has not been tried.
- **Dynamic leases** exist only in the simulator.
- **The host.** A UART host that pipelines commands across lanes is needed
  before lanes show their throughput on the board.

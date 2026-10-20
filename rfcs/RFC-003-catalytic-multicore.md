# RFC-003: Catalytic Multicore Scheduler and Multilane Processor

**Status:** Draft, implementation in progress (see *Results by phase*)  
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

*Implemented.* Results are recorded below as each phase lands.

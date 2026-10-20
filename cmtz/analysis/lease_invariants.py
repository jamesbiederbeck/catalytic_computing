"""Independent invariant checker for multilane lease traces (RFC-003).

Reads only the event trace, a list of plain dicts (`t`, `ev`, `lane`, ...),
and replays the lease bookkeeping on its own. It deliberately imports
nothing from `cmtz.isa.multicore`, so a bug in the scheduler cannot hide
itself by sharing code with its checker (the same rule as
`cmtz/projection_oracle.py`).

Checked:
  - a physical register is held by at most one borrower at a time, and a
    poisoned register is not leased again before it is cleared
  - one outstanding lease per lane, at most 8 registers, owner != borrower,
    no lane lends and borrows at once
  - a lease is released or poisoned only after its borrower's job ended:
    released (one cycle after the end) if the job halted (status 2),
    poisoned otherwise; never both, never left open at the end of the trace
  - with snapshots: the values at release equal the values at acquire
  - with access events: no lane touches a register leased to another lane
    or a poisoned one
  - every stall is resumed or aborted, and the cycles a job spent stalled
    add up to what its end event reports
"""

from __future__ import annotations

REGS_PER_LANE = 32
LEASE_MAX = 8
HALTED = 2


def check_trace(trace: list[dict]) -> list[str]:
    errs: list[str] = []
    held: dict[int, int] = {}            # physical -> borrower
    poisoned: set[int] = set()
    leases: dict[int, dict] = {}         # borrower -> acquire event
    running: dict[int, str] = {}         # lane -> job
    ended: dict[int, dict] = {}          # lane -> its last end event
    stalled: dict[int, int] = {}         # lane -> cycle the stall began
    stall_sum: dict[int, int] = {}
    last_t = -1

    def owner_of(phys: int) -> int:
        return phys // REGS_PER_LANE

    for e in trace:
        t, ev, lane = e["t"], e["ev"], e["lane"]
        if t < last_t and ev not in ("dispatch",):
            errs.append(f"t={t}: {ev} goes back in time (after t={last_t})")
        last_t = max(last_t, t)
        if ev == "acquire":
            phys = e["phys"]
            if lane in leases:
                errs.append(f"t={t}: lane {lane} acquires a second lease")
            if not 0 < len(phys) <= LEASE_MAX:
                errs.append(f"t={t}: lease of {len(phys)} registers")
            if len(set(phys)) != len(phys):
                errs.append(f"t={t}: lease names a register twice")
            if any(owner_of(i) != e["owner"] for i in phys):
                errs.append(f"t={t}: lease registers outside owner lane {e['owner']}")
            if e["owner"] == lane:
                errs.append(f"t={t}: lane {lane} borrows from itself")
            if e["owner"] in leases:
                errs.append(f"t={t}: owner lane {e['owner']} is borrowing")
            if any(owner_of(i) == lane for i in held):
                errs.append(f"t={t}: lane {lane} borrows while lending")
            for i in phys:
                if i in held:
                    errs.append(f"t={t}: register {i} leased to {held[i]} "
                                f"and to {lane}")
                if i in poisoned:
                    errs.append(f"t={t}: poisoned register {i} leased again")
                held[i] = lane
            leases[lane] = e
        elif ev in ("release", "poison"):
            a = leases.pop(lane, None)
            if a is None:
                errs.append(f"t={t}: {ev} without a lease on lane {lane}")
                continue
            end = ended.get(lane)
            if end is None or lane in running:
                errs.append(f"t={t}: {ev} while lane {lane}'s job runs")
            elif t != end["t"] + 1:
                errs.append(f"t={t}: {ev} not one cycle after the end at "
                            f"{end['t']}")
            elif (ev == "release") != (end["status"] == HALTED):
                errs.append(f"t={t}: {ev} after a job that ended with "
                            f"status {end['status']}")
            for i in a["phys"]:
                held.pop(i, None)
                if ev == "poison":
                    poisoned.add(i)
            if ev == "release" and "values" in a and "values" in e:
                if a["values"] != e["values"]:
                    errs.append(f"t={t}: lane {lane} returns {e['values']}, "
                                f"borrowed {a['values']}")
        elif ev == "cancel":
            a = leases.pop(lane, None)
            if a is None or lane in running:
                errs.append(f"t={t}: cancel of lane {lane}'s lease while it "
                            "runs or without one")
            elif ended.get(lane) and ended[lane]["t"] > a["t"]:
                errs.append(f"t={t}: cancel after lane {lane} ran with the lease")
            for i in (a or {}).get("phys", []):
                held.pop(i, None)
        elif ev == "clear_poison":
            for r in e["regs"]:
                poisoned.discard(lane * REGS_PER_LANE + r)
        elif ev == "dispatch":
            if lane in running:
                errs.append(f"t={t}: lane {lane} dispatched while running")
            running[lane] = e["job"]
            stall_sum[lane] = 0
        elif ev == "end":
            if running.pop(lane, None) != e["job"]:
                errs.append(f"t={t}: end of {e['job']} which lane {lane} "
                            "was not running")
            if lane in stalled:
                errs.append(f"t={t}: lane {lane} ends while stalled")
            if stall_sum.get(lane, 0) != e["stalls"]:
                errs.append(f"t={t}: {e['job']} reports {e['stalls']} stall "
                            f"cycles, the trace has {stall_sum.get(lane, 0)}")
            ended[lane] = e
        elif ev == "stall":
            stalled[lane] = t
            if not any(i in held and held[i] != lane or i in poisoned
                       for i in e["phys"]):
                errs.append(f"t={t}: lane {lane} stalls on {e['phys']}, "
                            "which nobody else holds")
        elif ev == "resume":
            s = stalled.pop(lane, None)
            if s is None or t - s != e["stalled"]:
                errs.append(f"t={t}: resume of lane {lane} does not match "
                            "its stall")
            else:
                stall_sum[lane] = stall_sum.get(lane, 0) + e["stalled"]
        elif ev == "poison_abort":
            s = stalled.pop(lane, None)
            if s is None:
                errs.append(f"t={t}: abort of lane {lane}, which was not stalled")
            else:
                stall_sum[lane] = stall_sum.get(lane, 0) + t - s
        elif ev == "access":
            for i in e["phys"]:
                if i in poisoned:
                    errs.append(f"t={t}: lane {lane} touches poisoned {i}")
                elif i in held and held[i] != lane:
                    errs.append(f"t={t}: lane {lane} touches {i}, leased "
                                f"to lane {held[i]}")
                elif owner_of(i) != lane and held.get(i) != lane:
                    errs.append(f"t={t}: lane {lane} touches {i} of lane "
                                f"{owner_of(i)} without a lease")
    for lane in leases:
        errs.append(f"lane {lane}'s lease is never released or poisoned")
    return errs

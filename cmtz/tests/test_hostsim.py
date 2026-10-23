"""SimCore: the cmtz-core UART protocol (host.py), step build included, in software."""

from pathlib import Path

import pytest

from cmtz.backends.isa_backend import emit_isa
from cmtz.compiler import compile_program
from cmtz.isa import CoreConfig, Machine
from cmtz.isa import sim as S
from cmtz.isa.demos import catalytic_loop
from cmtz.isa.host import PAUSED, CoreLink, info_frame
from cmtz.isa.hostsim import SimCore

CORE = CoreConfig(17, 19)
EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"


class Loop:
    """The port CoreLink wants, wired straight to a SimCore."""

    def __init__(self, sim: SimCore):
        self.sim, self.out = sim, bytearray()

    def write(self, data: bytes) -> None:
        self.out += self.sim.feed(data)

    def read(self, n: int) -> bytes:
        chunk = bytes(self.out[:n])
        del self.out[:n]
        return chunk


def rig(step: bool = True, slice: int = 64):
    sim = SimCore(CORE, step=step, slice=slice)
    return sim, CoreLink(Loop(sim), CORE, timeout=0.05)


def image(name: str, **kw):
    r = compile_program((EXAMPLES / name).read_text(), backend="python")
    assert r.ok, r.errors
    return emit_isa(r.ir_program, CORE, **kw)


def twin(link, img, inputs=None):
    """A Machine in the board's state, with the host's writes applied."""
    m = Machine(CORE, img)
    m.R, m.X = list(link.regs()[0]), list(link.regs()[1])
    m.D = link.read_data(0, CORE.dmem_size)
    if inputs:
        m.set_inputs(inputs)
    return m


def same(link, m, base=0):
    st = link.status()
    R, X = link.regs()
    want = PAUSED if m.status == S.RUNNING else m.status
    assert (st.status, st.error, st.retired, st.pc, st.interrupts - base) == \
           (want, m.error, m.retired, m.pc, len(m.dispatches))
    assert R == m.R and X == m.X


def test_info_frame_is_the_cores():
    _, link = rig()
    link.check_core()
    assert link.info() == info_frame(CORE, None)


def test_run_matches_the_simulator():
    b = image("fibonacci_pure.cmtz")
    _, link = rig()
    inputs = {k: 3 for k in b.inputs}
    m = twin(link, b.image, inputs)
    st = link.run(b.image, inputs)
    m.run()
    assert (st.status, st.retired, st.pc) == (m.status, m.retired, m.pc)
    assert link.outputs(b.image) == m.outputs()


def test_every_step_matches_the_simulator():
    b = image("fibonacci_pure.cmtz")
    _, link = rig()
    inputs = {k: 5 for k in b.inputs}
    link.load(b.image)
    for k, v in inputs.items():
        link.write_reg(b.image.inputs[k], v)
    m = twin(link, b.image, inputs)
    st = link.pause()
    assert (st.status, st.retired, st.pc) == (PAUSED, 0, b.image.entry)
    m.start()
    same(link, m)
    steps = 0
    while m.status == S.RUNNING:
        link.step()
        m.step()
        steps += 1
        same(link, m)
    assert steps == m.retired and m.status == S.HALTED
    assert link.outputs(b.image) == m.outputs()


def test_resume_continues_without_resetting():
    b = image("fibonacci.cmtz", matrix_dims={"fib_matrix": 3})
    _, link = rig()
    mat = {"fib_matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
    link.load(b.image)
    link.write_data(b.image.matrices_in["fib_matrix"].addr, sum(mat["fib_matrix"], []))
    m = twin(link, b.image)
    m.write_matrix("fib_matrix", mat["fib_matrix"])
    link.pause()
    m.start()
    for _ in range(40):
        link.step()
        m.step()
    same(link, m)
    link.resume()
    link.wait()
    while m.status == S.RUNNING:
        m.step()
    same(link, m)
    assert link.read_data(0, CORE.dmem_size) == m.D


def test_interrupt_while_paused_on_waiti():
    b = image("05_interrupt.cmtz")
    _, link = rig()
    inputs = {k: 2 for k in b.inputs}
    link.load(b.image)
    for k, v in inputs.items():
        link.write_reg(b.image.inputs[k], v)
    m = twin(link, b.image, inputs)
    link.pause()
    m.start()
    while not m.waiting:
        link.step()
        m.step()
    for _ in range(3):                      # waiting retires nothing
        link.step()
        m.step()
        same(link, m)
    link.trigger()
    m.pending = True
    while m.status == S.RUNNING:
        link.step()
        m.step()
        same(link, m)
    assert len(m.dispatches) == 1


def test_pause_in_the_middle_of_a_run():
    img = catalytic_loop(outer=40, inner=20)
    _, link = rig(slice=16)
    link.load(img)
    link.write_reg(img.inputs["x"], 6)
    m0 = twin(link, img, {"x": 6})
    full = Machine(CORE, img)
    full.R, full.X, full.D = list(m0.R), list(m0.X), list(m0.D)
    full.run()
    link.go()
    st = link.pause()
    assert st.status == PAUSED and 0 < st.retired < full.retired
    m = m0
    m.start()
    for _ in range(st.retired):
        m.step()
    same(link, m)                            # where it stopped is exactly the simulator's state
    link.resume()
    link.wait()
    while m.status == S.RUNNING:
        m.step()
    same(link, m)
    assert m.retired == full.retired


def test_refusals_and_what_works_while_paused():
    b = image("fibonacci_pure.cmtz")
    _, link = rig()
    with pytest.raises(RuntimeError):
        link.pause()                         # nothing loaded
    link.load(b.image)
    link.pause()
    with pytest.raises(RuntimeError):
        link.write_reg(1, 0)                 # W refused while paused
    link.read_data(0, 4)                     # M, S and Q work
    link.regs()
    link.pause()                             # asking again is fine
    link.step()
    link.abort()
    st = link.status()
    assert (st.status, st.error) == (S.ERROR, 7)
    with pytest.raises(RuntimeError):
        link.step()                          # an errored core is not paused
    assert link.pause().status == PAUSED     # but H pauses it at the entry again
    assert link.status().retired == 0


def test_a_plain_core_ignores_h_and_n():
    b = image("fibonacci_pure.cmtz")
    _, link = rig(step=False)
    link.load(b.image)
    with pytest.raises(TimeoutError):
        link.pause()                         # the board's behaviour on a core without a debugger
    st = link.run(b.image, {k: 1 for k in b.inputs})
    assert st.status == S.HALTED


def test_interrupt_count_and_last_dispatch_persist_between_runs():
    b = image("05_interrupt.cmtz")
    _, link = rig()
    inputs = {k: 4 for k in b.inputs}
    link.load(b.image)
    seen = []
    for expected in (1, 2):
        for k, v in inputs.items():
            link.write_reg(b.image.inputs[k], v)
        link.go()
        link.trigger()
        st = link.wait()
        assert st.interrupts == expected     # the count keeps growing across runs, like the board's
        seen.append((st.disp_retired, st.disp_pc))
    assert seen[0] == seen[1]                # the same program is interrupted at the same place
    m = Machine(CORE, b.image)
    m.set_inputs(inputs)
    m.run(irq_at=(0,))
    assert seen[0] == m.dispatches[-1]


# ── golden bytes for the browser client (visualizer/core.js) ────────────────

GOLDEN = Path(__file__).resolve().parent.parent.parent / "visualizer" / "testdata" / "core-golden.json"


def _golden() -> dict:
    """Encoder output and one recorded session, in hex, for core.test.mjs."""
    import json

    from cmtz.isa import host

    b = image("fibonacci_pure.cmtz")
    img = b.image
    sim = SimCore(CORE, step=True)
    session = []

    def ask(data: bytes) -> None:
        session.append({"send": data.hex(), "reply": sim.feed(data).hex()})

    ask(host.cmd_info())
    ask(host.cmd_load(img))
    ask(host.cmd_write_reg(img.inputs["embed_0"], 3))
    ask(host.cmd_write_reg(img.inputs["embed_1"], 4))
    ask(host.cmd_write_data(1000, [1, 2, 3]))
    ask(b"H")
    ask(b"Q")
    ask(b"S")
    for _ in range(2):
        ask(b"N")
        ask(b"Q")
        ask(b"S")
    ask(host.cmd_read_data(1000, 3))
    ask(b"T")
    ask(b"G")
    ask(b"Q")
    ask(b"N")                                # refused: nothing is paused any more
    ask(b"X")
    return {
        "image": {"words": img.words, "entry": img.entry, "handler": img.handler,
                  "inputs": img.inputs, "outputs": img.outputs},
        "encoders": {
            "load": host.cmd_load(img).hex(),
            "write_reg": host.cmd_write_reg(7, 12).hex(),
            "write_data": host.cmd_write_data(300, [1, 2, 18]).hex(),
            "read_data": host.cmd_read_data(1000, 3).hex(),
            "status": host.Status(4, 0, 70000, 3, 12, 34, 56).frame().hex(),
        },
        "session": session,
    }


def test_golden_for_the_browser_client(tmp_path):
    import json
    import os

    want = _golden()
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(want, indent=1) + "\n")
    assert json.loads(GOLDEN.read_text()) == want, "run with UPDATE_GOLDEN=1 to regenerate"

"""Tests for examples/fpga/build_projects.py (project generation, no board)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bp():
    spec = importlib.util.spec_from_file_location(
        "build_projects", ROOT / "examples" / "fpga" / "build_projects.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built(bp, tmp_path_factory):
    out = tmp_path_factory.mktemp("projects")
    bp.build(out, only=["phasor-basics", "fibonacci"])
    return out


def test_every_lowerable_example_is_a_target(bp):
    slugs = {slug for _, slug in bp.targets()}
    assert len(slugs) == 13 and "fibonacci" in slugs
    assert "cost-chain" not in slugs                 # matpow
    # a new example must be run on the board before it is listed
    assert {f"cmtz-{s}" for s in slugs} <= bp.ON_BOARD
    assert bp.ALU_VERIFIED <= bp.ON_BOARD


def test_noalu_is_the_default_and_alu_mapping_needs_verification(bp, built):
    assert "yosys-extra-options = -noalu" in (built / "cmtz-phasor-basics" / "apio.ini").read_text()
    assert "yosys-extra-options" not in (built / "cmtz-fibonacci" / "apio.ini").read_text()
    assert bp.uses_noalu("cmtz-some-new-example")
    assert bp.uses_noalu("cmtz-phasor-basics") and not bp.uses_noalu("cmtz-fibonacci")


def test_dropping_a_project_from_the_verified_set_turns_noalu_on(bp, tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "ALU_VERIFIED", set())
    bp.build(tmp_path, only=["fibonacci"])
    assert "yosys-extra-options = -noalu" in (tmp_path / "cmtz-fibonacci" / "apio.ini").read_text()


def test_readme_documents_the_flag(built):
    assert "## Build flag" in (built / "cmtz-phasor-basics" / "README.md").read_text()
    assert "## Build flag" not in (built / "cmtz-fibonacci" / "README.md").read_text()


def test_board_claim_is_tied_to_the_measured_set(bp, tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "ON_BOARD", set())
    bp.build(tmp_path, only=["fibonacci"])
    text = (tmp_path / "cmtz-fibonacci" / "README.md").read_text()
    assert "Not loaded on the board." in text and "Checked on the board" not in text


def test_debug_variants_follow_the_same_rule(bp, tmp_path):
    bp.build(tmp_path, debug=True, only=["fibonacci", "phasor-interference", "rotate-fuse"])
    ini = lambda n: (tmp_path / n / "apio.ini").read_text()
    assert "-noalu" not in ini("cmtz-fibonacci-debug")          # verified without the flag
    assert "-noalu" in ini("cmtz-phasor-interference-debug")    # needed
    assert "-noalu" in ini("cmtz-rotate-fuse-debug")            # debug build never run: default to -noalu
    d = tmp_path / "cmtz-fibonacci-debug"
    assert (d / "cmtz_fibonacci_dbg_uart.v").exists()
    assert "`S` (0x53)" in (d / "README.md").read_text()


# ── diagram.svg ──────────────────────────────────────────────────────────────

import xml.dom.minidom

from cmtz.compiler import compile_program
from cmtz.backends.verilog_backend import emit_verilog


@pytest.fixture(scope="module")
def all_built(bp, tmp_path_factory):
    out = tmp_path_factory.mktemp("all_projects")
    bp.build(out)
    return out


def _check_svg_rules(text: str):
    dom = xml.dom.minidom.parseString(text)             # parses as XML
    svg = dom.documentElement
    assert svg.tagName == "svg"
    w, h = int(svg.getAttribute("width")), int(svg.getAttribute("height"))
    assert svg.getAttribute("viewBox") == f"0 0 {w} {h}"
    assert svg.getAttribute("role") == "img" and len(svg.getAttribute("aria-label")) > 40
    kids = [n for n in svg.childNodes if n.nodeType == n.ELEMENT_NODE]
    assert kids[0].tagName == "title" and kids[0].firstChild.data
    bg = kids[1]                                         # its own white background
    assert (bg.tagName, bg.getAttribute("fill")) == ("rect", "#ffffff")
    assert (bg.getAttribute("width"), bg.getAttribute("height")) == (str(w), str(h))
    for banned in ("<style", "class=", "<script", "<image", "<foreignObject",
                   "paint-order", "href=", "@import", "url("):
        assert banned not in text, banned
    assert text.count("font-family=") == text.count("<text")
    assert "monospace" in text
    return w, h


def test_every_project_has_a_valid_diagram_embedded_in_its_readme(all_built):
    dirs = sorted(d for d in all_built.iterdir() if d.is_dir())
    assert len(dirs) == 13
    for d in dirs:
        w, h = _check_svg_rules((d / "diagram.svg").read_text())
        assert w <= 1400 and h <= 800, (d.name, w, h)     # readable when scaled
        assert "](diagram.svg)" in (d / "README.md").read_text(), d.name


def test_diagram_shows_the_datapath(all_built):
    svg = (all_built / "cmtz-fibonacci" / "diagram.svg").read_text()
    for needle in ("UART rx", "UART tx", "pin 70", "pin 69", "embed_0", "F12",
                   "stage 10", "delay 1", "req byte 0", "reply byte 0"):
        assert needle in svg, needle


def test_dead_nodes_drawn_only_in_debug_builds(bp, tmp_path):
    bp.build(tmp_path, only=["conjugate-catalytic"])
    bp.build(tmp_path, debug=True, only=["conjugate-catalytic"])
    plain = (tmp_path / "cmtz-conjugate-catalytic" / "diagram.svg").read_text()
    dbg = (tmp_path / "cmtz-conjugate-catalytic-debug" / "diagram.svg").read_text()
    assert "rot_rot_embed_0_embed_0_embed_2" not in plain      # no output depends on it
    assert "rot_rot_embed_0_embed_0_embed_2" in dbg            # kept alive by the state bus
    assert "cmds I R S" in dbg and "cmds I R S" not in plain


def test_multiplier_diagram(bp):
    import diagram
    src = "field(17, 293); embed(0,1); embed(1,5); mul(embed_0, embed_1) as m; measure(m, mod_p) as product;"
    b = emit_verilog(compile_program(src).ir_program, module_name="cmtz_mul")
    svg = diagram.draw_datapath(b, "cmtz-mul", "F_17 datapath")
    _check_svg_rules(svg)
    assert "mul" in svg and "product" in svg and "req byte 1" in svg


# ── cmtz-core ────────────────────────────────────────────────────────────────

def test_core_project(bp, tmp_path):
    d = bp.build_core(tmp_path)
    assert d.name == "cmtz-core"
    ini = (d / "apio.ini").read_text()
    assert "top-module = cmtz_core_uart" in ini and "-noalu" in ini
    for f in ("cmtz_core.v", "cmtz_core_tb.v", "cmtz_core_uart.v",
              "cmtz_core_uart_tb.v", "pins.cst", "README.md"):
        assert (d / f).exists(), f
    readme = (d / "README.md").read_text()
    if "cmtz-core" in bp.ON_BOARD:
        assert "Checked on the board" in readme
    else:
        assert "Not loaded on the board." in readme

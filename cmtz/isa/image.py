"""Program image: the one artifact the simulator, RTL testbenches and the
UART host tool all consume.

A `CoreConfig` describes a synthesised core (field and memory sizes; fixed
per bitstream). A `ProgramImage` is a program built for one such core plus
the maps a host needs to drive it: which registers are inputs and outputs,
and where runtime matrices live in data memory.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass, field

from .encoding import (
    BRANCH_OPS, FORMATS, ISA_VERSION, NUM_R, NUM_X, OPS, decode,
)


@dataclass(frozen=True)
class CoreConfig:
    p: int
    q: int
    pmem_size: int = 512
    dmem_size: int = 1024
    isa_version: int = ISA_VERSION

    @property
    def rw(self) -> int:
        """Bits per field register / data word (values up to q - 1)."""
        return max(1, (self.q - 1).bit_length())

    @property
    def core_hash(self) -> int:
        key = f"{self.isa_version}:{self.p}:{self.q}:{self.pmem_size}:{self.dmem_size}"
        return zlib.crc32(key.encode())


@dataclass
class MatrixSlot:
    addr: int
    n: int          # n x n, row-major


@dataclass
class ProgramImage:
    p: int
    q: int
    words: list[int]
    entry: int = 0
    handler: int | None = None
    isa_version: int = ISA_VERSION
    inputs: dict[str, int] = field(default_factory=dict)       # name -> R
    outputs: dict[str, int] = field(default_factory=dict)      # name -> R
    matrices_in: dict[str, MatrixSlot] = field(default_factory=dict)
    matrices_out: dict[str, MatrixSlot] = field(default_factory=dict)
    borrowed: list[int] = field(default_factory=list)          # handler
    handler_scratch: list[int] = field(default_factory=list)
    listing: list[str] = field(default_factory=list)

    @property
    def crc(self) -> int:
        """crc32 over the header fields the core sees plus every word."""
        h = [self.isa_version, self.p, self.q, len(self.words), self.entry,
             0xFFFF if self.handler is None else self.handler]
        data = b"".join(v.to_bytes(4, "big") for v in h + self.words)
        return zlib.crc32(data)

    def validate(self, core: CoreConfig) -> None:
        """Raise ValueError if this image cannot run on `core`."""
        errs = []
        if self.isa_version != core.isa_version:
            errs.append(f"ISA version {self.isa_version}, core has {core.isa_version}")
        if (self.p, self.q) != (core.p, core.q):
            errs.append(f"field ({self.p}, {self.q}), core has ({core.p}, {core.q})")
        n = len(self.words)
        if not 0 < n <= core.pmem_size:
            errs.append(f"{n} words, program memory holds {core.pmem_size}")
        for what, pc in (("entry", self.entry), ("handler", self.handler)):
            if pc is not None and not 0 <= pc < n:
                errs.append(f"{what} {pc} outside the program")
        for i, w in enumerate(self.words):
            try:
                ins = decode(w)
            except ValueError as e:
                errs.append(f"word {i}: {e}")
                continue
            for kind, fld in FORMATS[OPS[ins.op].fmt]:
                v = getattr(ins, fld if kind != "m" else "b")
                if kind == "r" and v >= NUM_R or kind in "xm" and v >= NUM_X:
                    errs.append(f"word {i}: register {v} out of range")
            if ins.op in BRANCH_OPS and ins.imm >= n:
                errs.append(f"word {i}: jump target {ins.imm} outside the program")
            if ins.op in ("MULI",) and ins.imm >= core.p:
                errs.append(f"word {i}: MULI immediate {ins.imm} >= p")
            if ins.op in ("LDI", "HORNER") and ins.imm >= core.q:
                errs.append(f"word {i}: immediate {ins.imm} >= q")
        for name, r in {**self.inputs, **self.outputs}.items():
            if not 0 <= r < NUM_R:
                errs.append(f"register map {name!r}: R{r} out of range")
        for name, s in {**self.matrices_in, **self.matrices_out}.items():
            if s.addr < 0 or s.addr + s.n * s.n > core.dmem_size:
                errs.append(f"matrix {name!r} outside data memory")
        if errs:
            raise ValueError("invalid program image:\n  " + "\n  ".join(errs))

    # ── serialisation ────────────────────────────────────────────────

    def to_json(self) -> str:
        d = asdict(self)
        d["crc"] = self.crc
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, text: str) -> ProgramImage:
        d = json.loads(text)
        crc = d.pop("crc", None)
        for k in ("matrices_in", "matrices_out"):
            d[k] = {n: MatrixSlot(**s) for n, s in d[k].items()}
        img = cls(**d)
        if crc is not None and crc != img.crc:
            raise ValueError(f"image crc mismatch: file {crc:#x}, "
                             f"contents {img.crc:#x}")
        return img

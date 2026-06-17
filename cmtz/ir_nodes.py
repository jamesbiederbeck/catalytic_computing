"""IR node definitions for Cook-Mertz v4.

Every IR node carries:
  - field: IRField (arithmetic context, never None after elaboration)
  - Cost model fields (Lemma 2.1): recursive_calls, basic_instructions, num_registers
  - Catalytic annotation: is_catalytic flag

v4 additions (phasor network support):
  - IRAdd: additive superposition gate
  - IRComplexEmbed: initialize a complex F_{p²} register
  - IRConjugate: complex conjugate
  - IRMagnitudeSq: |z|² → F_p  (type-lowers from F_{p²})

v1 bugs fixed:
  - IRRotate.theta: float → IRRotate.j: int  (stay in modular arithmetic)
  - IRPoly has no field → IRPoly.field: IRField  (bigint safe)
  - parents: List[IRNode] = [] → field(default_factory=list) (mutable default)
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from .field import IRField, primitive_roots_Fp, to_base


# ── Base IR node ─────────────────────────────────────────────────────────────

@dataclass
class IRNode:
    """Base class for all IR DAG nodes."""
    name: str
    parents: list[IRNode] = field(default_factory=list)  # v1 bug fixed
    ir_field: Optional[IRField] = None

    # Cost model (Lemma 2.1)
    recursive_calls: int = 0
    basic_instructions: int = 0
    num_registers: int = 0

    # Catalytic annotation
    is_catalytic: bool = False

    def __hash__(self):
        return id(self)


# ── Concrete IR nodes ────────────────────────────────────────────────────────

@dataclass
class IREmbed(IRNode):
    """Initialize a register with a phase offset.

    psi is an integer exponent: physical phase = 2π·psi/m.
    In F_p arithmetic: register ← ω^psi.
    """
    index: int = 0
    psi: int = 0

    def __post_init__(self):
        self.basic_instructions = 1
        self.num_registers = 1


@dataclass
class IRPrimitiveRoots(IRNode):
    """Compute the root table {ω^0, ω^1, ..., ω^(m-1)} in F_p.

    Replaces v1's cyclo(n) which called sympy.cyclotomic_poly() —
    the minimal polynomial Φ_n(x), a completely different object.

    The root table is computed once and referenced by IRRotate nodes.
    """
    order: int = 0
    roots: tuple[int, ...] = ()

    @classmethod
    def build(cls, ir_field: IRField, name: str) -> IRPrimitiveRoots:
        roots = primitive_roots_Fp(ir_field.p)
        node = cls(
            name=name,
            ir_field=ir_field,
            order=ir_field.m,
            roots=roots,
        )
        node.basic_instructions = ir_field.m  # one multiply per root
        node.num_registers = ir_field.m
        return node


@dataclass
class IRRotate(IRNode):
    """Multiply a register value by ω^j in F_p.

    v1 used theta: float — ambiguous between complex phase and modular int.
    v2 uses j: int (exponent) so computation stays in exact modular arithmetic.
    Physical angle (for analog backend) is derived: θ = 2π·j/m.
    """
    j: int = 0
    src: Optional[IRNode] = None

    def __post_init__(self):
        self.basic_instructions = 1
        self.num_registers = 1

    @property
    def theta(self) -> float:
        """Physical phase offset for analog backend."""
        if self.ir_field is None:
            raise ValueError("IRRotate has no field annotation")
        return 2 * math.pi * self.j / self.ir_field.m

    def fused_with(self, other: IRRotate) -> IRRotate:
        """Merge adjacent rotations: ω^j1 · ω^j2 = ω^(j1+j2 mod m)."""
        if self.ir_field != other.ir_field:
            raise ValueError("Cannot fuse rotations from different fields")
        return IRRotate(
            name=f"{self.name}_fused",
            j=(self.j + other.j) % self.ir_field.m,
            ir_field=self.ir_field,
            src=self.src,
        )


@dataclass
class IRPoly(IRNode):
    """Evaluate a polynomial over F_q, then reduce mod p.

    coeffs are exact Python ints (arbitrary precision). No floats.
    coeffs[i] = coefficient of x^i (ascending order).

    v1 had no field annotation — coefficients could silently overflow
    when using torch.float64. v2 tags every IRPoly with its IRField
    and uses explicit mod q after each operation.
    """
    coeffs: list[int] = field(default_factory=list)
    degree: int = 0

    def __post_init__(self):
        if self.coeffs:
            self.degree = len(self.coeffs) - 1
        self.basic_instructions = max(1, self.degree)  # Horner steps
        self.num_registers = 2  # accumulator + temp


@dataclass
class IRMatPow(IRNode):
    """Matrix powering: compute M^d via Theorem 1.2.

    Uses base-δ decomposition (Lemma 3.12) where δ = 2^(3/ε).
    Each digit dispatches to polynomial evaluation over F_q.

    epsilon controls the recursive-call vs register tradeoff:
      - small ε → fewer recursive calls, more registers
      - large ε → more recursive calls, fewer registers
    """
    matrix_name: str = ""
    d: int = 0
    epsilon: float = 0.5
    n_dim: int = 0

    @property
    def delta(self) -> int:
        return max(2, int(2 ** (3 / self.epsilon)))

    def cost_bound(self) -> dict:
        """Theorem 1.2 cost bounds for verification."""
        d, delta, eps, n = self.d, self.delta, self.epsilon, self.n_dim
        log_d = math.log(d + 1)
        return {
            'recursive_calls': (d ** eps) * log_d,
            'basic_instructions': (n ** math.exp(1/eps)) * (d ** eps) * log_d,
            'registers_Fq': n ** math.exp(1/eps),
            'registers_Fp': n ** 2,
        }


@dataclass
class IRRegion(IRNode):
    """Advisory region — main-process use, no verification (§2.6).

    The main process owns its registers and may compute any
    roots-of-unity program over them. The catalytic_regs list
    documents intended workspace but carries no enforcement.
    Produced by the `catalytic {}` DSL keyword.
    """
    inner_nodes: list[IRNode] = field(default_factory=list)
    catalytic_regs: list[str] = field(default_factory=list)


@dataclass
class IRCatalyticRegion(IRNode):
    """Strict catalytic region — interrupt handler use, verified (§2.7).

    Represents a handler that borrows registers from a running
    computation. The verifier enforces that no inner node clobbers
    any register listed in catalytic_regs. Produced by `interrupt {}`.
    """
    inner_nodes: list[IRNode] = field(default_factory=list)
    catalytic_regs: list[str] = field(default_factory=list)


@dataclass
class IRCompose(IRNode):
    """Compose two sub-programs with Lemma 2.1 cost accounting.

    v1 just concatenated node lists. v2 correctly computes:
      recursive_calls    = f.recursive_calls × g.recursive_calls
      basic_instructions = f.basic_instructions + f.recursive_calls × g.basic_instructions
      num_registers      = max(f.num_registers, g.num_registers)
    """
    pass


@dataclass
class IRMeasure(IRNode):
    """Extract a scalar from a register.
    v2 adds 'mod_p' mode: return value reduced to F_p."""
    mode: str = "real"  # 'real' | 'imag' | 'mag' | 'trace' | 'mod_p'
    src: Optional[IRNode] = None


@dataclass
class IRCycloPhiPoly(IRNode):
    """The actual cyclotomic polynomial Φ_n(x) — kept for FIR filter use.
    This is what v1's cyclo() computed, but that was wrong for the ω-gate.
    Renamed to avoid confusion."""
    n: int = 0
    poly_coeffs: list[int] = field(default_factory=list)


# ── v4: Phasor network primitives ────────────────────────────────────────────

@dataclass
class IRAdd(IRNode):
    """Additive superposition gate: compute (a + b) mod p.

    For real fields (F_p): result = (a + b) % p
    For complex fields (F_{p²}): result = add_Fp2(a, b)
    Both parents must share the same IRField.
    """

    def __post_init__(self):
        self.basic_instructions = 1
        self.num_registers = 1


@dataclass
class IRComplexEmbed(IRNode):
    """Initialize a complex register with ω^re_psi + i·ω^im_psi in F_{p²}.

    Requires ir_field.is_complex == True.
    Name defaults to cembed_N in the lowerer.
    """
    re_psi: int = 0
    im_psi: int = 0

    def __post_init__(self):
        self.basic_instructions = 1
        self.num_registers = 1


@dataclass
class IRConjugate(IRNode):
    """Complex conjugate: a+bi → a-bi mod p.

    Requires ir_field.is_complex == True.
    Input (single parent) must also be in a complex field.
    """

    def __post_init__(self):
        self.basic_instructions = 1
        self.num_registers = 1


@dataclass
class IRMagnitudeSq(IRNode):
    """Magnitude squared: |a+bi|² = a² - b²·c mod p.

    Type-lowers from F_{p²} to F_p.
    Input ir_field must be complex; the output node's ir_field
    should be set to ir_field.real_field() by the lowerer.
    """

    def __post_init__(self):
        self.basic_instructions = 3   # two squares + one add + mod p
        self.num_registers = 2


# ── IR Program container ─────────────────────────────────────────────────────

class IRProgram:
    """Container for the full IR DAG with topological ordering guarantee.

    v1 used a plain dict with no ordering guarantee. v2 maintains
    insertion order and provides topo_order() for correct evaluation.
    """

    def __init__(self, ir_field: Optional[IRField] = None):
        self.ir_field = ir_field
        self._nodes: OrderedDict[str, IRNode] = OrderedDict()
        self._topo_sorted: bool = False

    def add_node(self, node: IRNode) -> None:
        if node.name in self._nodes:
            raise ValueError(f"Duplicate node name: {node.name}")
        self._nodes[node.name] = node
        self._topo_sorted = False

    def get_node(self, name: str) -> IRNode:
        if name not in self._nodes:
            raise KeyError(f"Unknown node: {name}")
        return self._nodes[name]

    def nodes(self) -> list[tuple[str, IRNode]]:
        return list(self._nodes.items())

    def topo_order(self) -> list[tuple[str, IRNode]]:
        """Return nodes in topological order. Raises if cycles exist."""
        visited: set[str] = set()
        in_stack: set[str] = set()
        order: list[str] = []

        def visit(name: str):
            if name in in_stack:
                raise CycleError(f"Cycle detected involving node '{name}'")
            if name in visited:
                return
            in_stack.add(name)
            node = self._nodes[name]
            for parent in node.parents:
                if parent.name in self._nodes:
                    visit(parent.name)
            in_stack.discard(name)
            visited.add(name)
            order.append(name)

        for name in self._nodes:
            visit(name)

        self._topo_sorted = True
        return [(n, self._nodes[n]) for n in order]

    def __len__(self) -> int:
        return len(self._nodes)


class CycleError(Exception):
    pass

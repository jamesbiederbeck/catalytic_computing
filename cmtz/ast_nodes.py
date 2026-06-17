"""AST node definitions for Cook-Mertz DSL v4.

v2 syntax changes from v1:
  - field(p, q) declaration for arithmetic context
  - roots(p) replaces cyclo(n) — computes primitive roots, not Φ_n
  - cyclo_phi(n) retained for actual cyclotomic polynomial use
  - rotate j parameter is integer exponent, not float cost
  - matpow(M, d, epsilon) for matrix powering
  - catalytic { ... } restoring (...) for catalytic regions
  - measure gains 'mod_p' mode

v3 syntax changes:
  - interrupt { ... } restoring (...) for strict catalytic regions

v4 syntax changes (phasor network support):
  - cfield(p) / cfield(p, q) for complex F_{p²} programs
  - add(a, b) as name — additive superposition
  - cembed(re_psi, im_psi) — complex register initialization
  - conj(a) as name — complex conjugate
  - magsq(a) as name — |z|² → F_p
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ASTNode:
    """Base class for all AST nodes. Carries source location."""
    line: int = 0
    col: int = 0


@dataclass
class FieldDecl(ASTNode):
    """field(p, q); — declares the arithmetic field for this program."""
    p: int = 0
    q: Optional[int] = None  # if None, q defaults to p


@dataclass
class CFieldDecl(ASTNode):
    """cfield(p) or cfield(p, q); — declares a complex F_{p²} field.
    The non-residue c is auto-computed by the elaborator."""
    p: int = 0
    q: Optional[int] = None  # if None, elaborator chooses q >= p²


@dataclass
class EmbedStmt(ASTNode):
    """embed(index, psi); — initialize a phase register."""
    index: int = 0
    psi: int = 0  # v2: integer phase exponent, not float


@dataclass
class RotateStmt(ASTNode):
    """rotate(src, dst, j); — apply ω^j rotation.
    v1 had cost: float; v2 uses j: int (exponent of primitive root)."""
    src: str = ""
    dst: str = ""
    j: int = 0


@dataclass
class ComposeStmt(ASTNode):
    """compose(id1, id2, ...); — compose pipeline stages."""
    pipeline: list[str] = field(default_factory=list)


@dataclass
class RootsStmt(ASTNode):
    """roots(p) as name; — compute primitive roots in F_p.
    Replaces v1's cyclo(n) which incorrectly computed Φ_n(x)."""
    p: int = 0
    name: str = ""


@dataclass
class CycloPhiStmt(ASTNode):
    """cyclo_phi(n) as name; — compute cyclotomic polynomial Φ_n(x).
    Renamed from v1's cyclo() to avoid confusion with root enumeration."""
    n: int = 0
    name: str = ""


@dataclass
class MatPowStmt(ASTNode):
    """matpow(M, d, epsilon) as name; — matrix powering via Thm 1.2."""
    matrix: str = ""
    d: int = 0
    epsilon: float = 0.0
    name: str = ""


@dataclass
class CatalyticStmt(ASTNode):
    """catalytic { body } restoring (reg1, reg2, ...);
    Advisory region. Lowers to IRRegion. No verification enforced.
    Main-process use: documents register intent only."""
    body: list[ASTNode] = field(default_factory=list)
    restoring: list[str] = field(default_factory=list)


@dataclass
class InterruptStmt(ASTNode):
    """interrupt { body } restoring (reg1, reg2, ...);
    Strict catalytic region. Lowers to IRCatalyticRegion. Verified.
    Handler use: listed registers must be provably restored on exit."""
    body: list[ASTNode] = field(default_factory=list)
    restoring: list[str] = field(default_factory=list)


@dataclass
class MeasureStmt(ASTNode):
    """measure(id, mode) as var; — extract scalar from register."""
    src: str = ""
    mode: str = ""  # 'real' | 'imag' | 'mag' | 'trace' | 'mod_p'
    name: str = ""


# ── v4: Phasor network statement nodes ───────────────────────────────────────

@dataclass
class AddStmt(ASTNode):
    """add(a, b) as name; — additive superposition gate."""
    a: str = ""
    b: str = ""
    name: str = ""


@dataclass
class CEmbedStmt(ASTNode):
    """cembed(re_psi, im_psi); — complex register initialization.
    Produces a register named cembed_N (N auto-incremented by lowerer)."""
    re_psi: int = 0
    im_psi: int = 0


@dataclass
class ConjStmt(ASTNode):
    """conj(a) as name; — complex conjugate."""
    src: str = ""
    name: str = ""


@dataclass
class MagSqStmt(ASTNode):
    """magsq(a) as name; — magnitude squared, type-lowers F_{p²} → F_p."""
    src: str = ""
    name: str = ""


@dataclass
class Program(ASTNode):
    """Top-level program: optional field declaration + statements."""
    field_decl: Optional[FieldDecl | CFieldDecl] = None
    statements: list[ASTNode] = field(default_factory=list)

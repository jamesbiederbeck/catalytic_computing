"""Cook-Mertz Compiler — spec revision v4

A compiler for the Cook-Mertz computation model DSL, targeting
exact modular arithmetic over finite fields F_p and F_{p²} with
catalytic register tracking, Theorem 1.2 cost verification, and
phasor network support (addition, complex registers, conjugate,
magnitude squared).

v4 additions:
  - IRAdd, IRComplexEmbed, IRConjugate, IRMagnitudeSq
  - IRField.c for F_{p²} complex extension
  - cfield(p) DSL declaration

Package version follows semver by DSL-source compatibility, which is a
different axis than the spec's v1/v2/v3/v4 revision labels above: v1->v2
renamed/changed DSL constructs (breaking); v2->v3 and v3->v4 were purely
additive ("all vN programs compile unchanged under vN+1", per the spec's
own migration tables). So spec v4 maps to semver 2.2.0, not 4.0.0 — see
Cook_Mertz_Spec_v2.md §1 for the full mapping table.
"""

__version__ = "2.2.0"
SPEC_REVISION = 4

from .field import IRField, primitive_roots_Fp, to_base, find_nonresidue
from .field import add_Fp2, conj_Fp2, magsq_Fp2
from .lexer import tokenize, Token
from .ast_nodes import *
from .ast_nodes import (
    InterruptStmt, CFieldDecl, AddStmt, CEmbedStmt, ConjStmt, MagSqStmt,
)
from .ir_nodes import *
from .parser import Parser
from .elaborator import Elaborator
from .lowering import lower
from .compiler import compile_program

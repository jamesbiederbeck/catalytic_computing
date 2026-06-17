"""Cook-Mertz Compiler v4

A compiler for the Cook-Mertz computation model DSL, targeting
exact modular arithmetic over finite fields F_p and F_{p²} with
catalytic register tracking, Theorem 1.2 cost verification, and
phasor network support (addition, complex registers, conjugate,
magnitude squared).

v4 additions:
  - IRAdd, IRComplexEmbed, IRConjugate, IRMagnitudeSq
  - IRField.c for F_{p²} complex extension
  - cfield(p) DSL declaration
"""

__version__ = "4.0.0"

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

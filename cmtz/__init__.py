"""Cook-Mertz Compiler v2

A compiler for the Cook-Mertz computation model DSL, targeting
exact modular arithmetic over finite fields F_p with catalytic
register tracking and Theorem 1.2 cost verification.

Key fixes over v1:
  - cyclo(n) replaced with primitive root enumeration in F_p
  - IRRotate uses integer exponent j, not float theta
  - IRPoly carries field annotation; bigint-safe
  - Catalytic register restore invariant is verified
  - Cost model implements Lemma 2.1 composition accounting
"""

__version__ = "2.0.0"

from .field import IRField, primitive_roots_Fp, to_base
from .lexer import tokenize, Token
from .ast_nodes import *
from .ir_nodes import *
from .parser import Parser
from .elaborator import Elaborator
from .lowering import lower
from .compiler import compile_program

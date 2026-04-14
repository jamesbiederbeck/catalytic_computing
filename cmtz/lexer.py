"""Lexer for Cook-Mertz DSL v2.

v1 bugs fixed:
  - TOKEN_SPEC had MODE after ID, so 'real'/'imag'/'mag'/'trace'
    tokenized as ID instead of MODE. Fixed by placing MODE and
    keywords before ID in the alternation.
  - Added line/col tracking for error reporting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

# ── Token types ──────────────────────────────────────────────────────────────
# ORDER MATTERS: longer/more-specific patterns must come before shorter ones.
# In particular, MODE keywords and statement keywords must precede ID,
# otherwise 'real', 'field', etc. match as generic identifiers.

TOKEN_SPEC: list[tuple[str, str]] = [
    # Whitespace and comments (skipped)
    ('SKIP',        r'[ \t]+'),
    ('NEWLINE',     r'\n'),
    ('COMMENT',     r'//[^\n]*|--[^\n]*'),

    # Literals
    ('FLOAT',       r'-?\d+\.\d+'),
    ('INT',         r'-?\d+'),

    # Modes — MUST be before ID
    ('MODE',        r'\b(?:real|imag|mag|trace|mod_p)\b'),

    # Keywords — MUST be before ID
    ('KW_FIELD',    r'\bfield\b'),
    ('KW_EMBED',    r'\bembed\b'),
    ('KW_ROTATE',   r'\brotate\b'),
    ('KW_COMPOSE',  r'\bcompose\b'),
    ('KW_MEASURE',  r'\bmeasure\b'),
    ('KW_ROOTS',    r'\broots\b'),
    ('KW_CYCLO_PHI', r'\bcyclo_phi\b'),
    ('KW_MATPOW',   r'\bmatpow\b'),
    ('KW_CATALYTIC', r'\bcatalytic\b'),
    ('KW_RESTORING', r'\brestoring\b'),
    ('KW_AS',       r'\bas\b'),

    # Identifiers (after keywords and modes)
    ('ID',          r'[A-Za-z_][A-Za-z0-9_]*'),

    # Punctuation
    ('LPAREN',      r'\('),
    ('RPAREN',      r'\)'),
    ('LBRACE',      r'\{'),
    ('RBRACE',      r'\}'),
    ('COMMA',       r','),
    ('SEMI',        r';'),

    # Catch-all for errors
    ('MISMATCH',    r'.'),
]

MASTER_RE = re.compile(
    '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPEC)
)


@dataclass(frozen=True)
class Token:
    """A token with type, value, and source location."""
    type: str
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"


def tokenize(source: str) -> list[Token]:
    """Tokenize a Cook-Mertz DSL v2 source string.

    Returns a list of Tokens with line/col info.
    Raises LexError on unrecognized characters.
    """
    tokens: list[Token] = []
    line = 1
    line_start = 0

    for mo in MASTER_RE.finditer(source):
        kind = mo.lastgroup
        value = mo.group()
        col = mo.start() - line_start + 1

        if kind == 'NEWLINE':
            line += 1
            line_start = mo.end()
        elif kind == 'SKIP' or kind == 'COMMENT':
            pass
        elif kind == 'MISMATCH':
            raise LexError(f"Unexpected character {value!r} at {line}:{col}")
        else:
            tokens.append(Token(type=kind, value=value, line=line, col=col))

    tokens.append(Token(type='EOF', value='', line=line, col=0))
    return tokens


class LexError(Exception):
    pass

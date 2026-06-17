"""Recursive-descent parser for Cook-Mertz DSL v2.

Produces a typed AST. Rejects unknown IDs and enforces literal
parameters where required.

Grammar:
  program      ::= field_decl? statement*
  field_decl   ::= 'field' '(' INT ',' INT ')' ';'
                 | 'field' '(' INT ')' ';'
  statement    ::= embed_stmt | rotate_stmt | compose_stmt
                 | measure_stmt | roots_stmt | cyclo_phi_stmt
                 | matpow_stmt | catalytic_stmt
"""

from __future__ import annotations

from .lexer import Token, LexError
from .ast_nodes import (
    Program, FieldDecl, CFieldDecl, EmbedStmt, RotateStmt, ComposeStmt,
    RootsStmt, CycloPhiStmt, MatPowStmt, CatalyticStmt, InterruptStmt,
    MeasureStmt, AddStmt, CEmbedStmt, ConjStmt, MagSqStmt, ASTNode,
)


class ParseError(Exception):
    def __init__(self, msg: str, token: Token | None = None):
        loc = f" at {token.line}:{token.col}" if token else ""
        super().__init__(f"{msg}{loc}")
        self.token = token


class Parser:
    """Recursive-descent parser for Cook-Mertz DSL v2."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # ── Helpers ──────────────────────────────────────────────────────────

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tok_type: str) -> Token:
        tok = self.peek()
        if tok.type != tok_type:
            raise ParseError(
                f"Expected {tok_type}, got {tok.type} ({tok.value!r})", tok
            )
        return self.advance()

    def expect_int(self) -> int:
        tok = self.expect('INT')
        return int(tok.value)

    def expect_float(self) -> float:
        tok = self.peek()
        if tok.type == 'FLOAT':
            self.advance()
            return float(tok.value)
        elif tok.type == 'INT':
            self.advance()
            return float(tok.value)
        raise ParseError(f"Expected number, got {tok.type}", tok)

    def expect_id(self) -> str:
        tok = self.expect('ID')
        return tok.value

    def at(self, tok_type: str) -> bool:
        return self.peek().type == tok_type

    def at_keyword(self, kw: str) -> bool:
        return self.peek().type == f'KW_{kw.upper()}'

    # ── Top-level ────────────────────────────────────────────────────────

    def parse(self) -> Program:
        """Parse a complete program."""
        loc = self.peek()
        field_decl = None
        if self.at_keyword('field'):
            field_decl = self._parse_field_decl()
        elif self.at_keyword('cfield'):
            field_decl = self._parse_cfield_decl()

        stmts = []
        while not self.at('EOF'):
            stmts.append(self._parse_statement())

        return Program(
            field_decl=field_decl,
            statements=stmts,
            line=loc.line,
            col=loc.col,
        )

    # ── Field declaration ────────────────────────────────────────────────

    def _parse_field_decl(self) -> FieldDecl:
        tok = self.advance()  # consume 'field'
        self.expect('LPAREN')
        p = self.expect_int()
        q = None
        if self.at('COMMA'):
            self.advance()
            q = self.expect_int()
        self.expect('RPAREN')
        self.expect('SEMI')
        return FieldDecl(p=p, q=q, line=tok.line, col=tok.col)

    def _parse_cfield_decl(self) -> CFieldDecl:
        tok = self.advance()  # consume 'cfield'
        self.expect('LPAREN')
        p = self.expect_int()
        q = None
        if self.at('COMMA'):
            self.advance()
            q = self.expect_int()
        self.expect('RPAREN')
        self.expect('SEMI')
        return CFieldDecl(p=p, q=q, line=tok.line, col=tok.col)

    # ── Statement dispatch ───────────────────────────────────────────────

    def _parse_statement(self) -> ASTNode:
        tok = self.peek()
        dispatch = {
            'KW_EMBED': self._parse_embed,
            'KW_ROTATE': self._parse_rotate,
            'KW_COMPOSE': self._parse_compose,
            'KW_MEASURE': self._parse_measure,
            'KW_ROOTS': self._parse_roots,
            'KW_CYCLO_PHI': self._parse_cyclo_phi,
            'KW_MATPOW': self._parse_matpow,
            'KW_CATALYTIC': self._parse_catalytic,
            'KW_INTERRUPT': self._parse_interrupt,
            'KW_ADD': self._parse_add,
            'KW_CEMBED': self._parse_cembed,
            'KW_CONJ': self._parse_conj,
            'KW_MAGSQ': self._parse_magsq,
        }
        handler = dispatch.get(tok.type)
        if handler is None:
            raise ParseError(f"Expected statement, got {tok.type} ({tok.value!r})", tok)
        return handler()

    # ── Individual statement parsers ─────────────────────────────────────

    def _parse_embed(self) -> EmbedStmt:
        tok = self.advance()  # 'embed'
        self.expect('LPAREN')
        index = self.expect_int()
        self.expect('COMMA')
        psi = self.expect_int()
        self.expect('RPAREN')
        self.expect('SEMI')
        return EmbedStmt(index=index, psi=psi, line=tok.line, col=tok.col)

    def _parse_rotate(self) -> RotateStmt:
        tok = self.advance()  # 'rotate'
        self.expect('LPAREN')
        src = self.expect_id()
        self.expect('COMMA')
        dst = self.expect_id()
        self.expect('COMMA')
        j = self.expect_int()
        self.expect('RPAREN')
        self.expect('SEMI')
        return RotateStmt(src=src, dst=dst, j=j, line=tok.line, col=tok.col)

    def _parse_compose(self) -> ComposeStmt:
        tok = self.advance()  # 'compose'
        self.expect('LPAREN')
        ids = [self.expect_id()]
        while self.at('COMMA'):
            self.advance()
            ids.append(self.expect_id())
        self.expect('RPAREN')
        self.expect('SEMI')
        return ComposeStmt(pipeline=ids, line=tok.line, col=tok.col)

    def _parse_roots(self) -> RootsStmt:
        tok = self.advance()  # 'roots'
        self.expect('LPAREN')
        p = self.expect_int()
        self.expect('RPAREN')
        self.expect('KW_AS')
        name = self.expect_id()
        self.expect('SEMI')
        return RootsStmt(p=p, name=name, line=tok.line, col=tok.col)

    def _parse_cyclo_phi(self) -> CycloPhiStmt:
        tok = self.advance()  # 'cyclo_phi'
        self.expect('LPAREN')
        n = self.expect_int()
        self.expect('RPAREN')
        self.expect('KW_AS')
        name = self.expect_id()
        self.expect('SEMI')
        return CycloPhiStmt(n=n, name=name, line=tok.line, col=tok.col)

    def _parse_matpow(self) -> MatPowStmt:
        tok = self.advance()  # 'matpow'
        self.expect('LPAREN')
        matrix = self.expect_id()
        self.expect('COMMA')
        d = self.expect_int()
        self.expect('COMMA')
        epsilon = self.expect_float()
        self.expect('RPAREN')
        self.expect('KW_AS')
        name = self.expect_id()
        self.expect('SEMI')
        return MatPowStmt(
            matrix=matrix, d=d, epsilon=epsilon, name=name,
            line=tok.line, col=tok.col,
        )

    def _parse_catalytic(self) -> CatalyticStmt:
        tok = self.advance()  # 'catalytic'
        self.expect('LBRACE')
        body = []
        while not self.at('RBRACE'):
            body.append(self._parse_statement())
        self.expect('RBRACE')
        self.expect('KW_RESTORING')
        self.expect('LPAREN')
        regs = [self.expect_id()]
        while self.at('COMMA'):
            self.advance()
            regs.append(self.expect_id())
        self.expect('RPAREN')
        self.expect('SEMI')
        return CatalyticStmt(
            body=body, restoring=regs, line=tok.line, col=tok.col,
        )

    def _parse_interrupt(self) -> InterruptStmt:
        tok = self.advance()  # 'interrupt'
        self.expect('LBRACE')
        body = []
        while not self.at('RBRACE'):
            body.append(self._parse_statement())
        self.expect('RBRACE')
        self.expect('KW_RESTORING')
        self.expect('LPAREN')
        regs = [self.expect_id()]
        while self.at('COMMA'):
            self.advance()
            regs.append(self.expect_id())
        self.expect('RPAREN')
        self.expect('SEMI')
        return InterruptStmt(
            body=body, restoring=regs, line=tok.line, col=tok.col,
        )

    def _parse_add(self) -> AddStmt:
        tok = self.advance()  # 'add'
        self.expect('LPAREN')
        a = self.expect_id()
        self.expect('COMMA')
        b = self.expect_id()
        self.expect('RPAREN')
        self.expect('KW_AS')
        name = self.expect_id()
        self.expect('SEMI')
        return AddStmt(a=a, b=b, name=name, line=tok.line, col=tok.col)

    def _parse_cembed(self) -> CEmbedStmt:
        tok = self.advance()  # 'cembed'
        self.expect('LPAREN')
        re_psi = self.expect_int()
        self.expect('COMMA')
        im_psi = self.expect_int()
        self.expect('RPAREN')
        self.expect('SEMI')
        return CEmbedStmt(re_psi=re_psi, im_psi=im_psi, line=tok.line, col=tok.col)

    def _parse_conj(self) -> ConjStmt:
        tok = self.advance()  # 'conj'
        self.expect('LPAREN')
        src = self.expect_id()
        self.expect('RPAREN')
        self.expect('KW_AS')
        name = self.expect_id()
        self.expect('SEMI')
        return ConjStmt(src=src, name=name, line=tok.line, col=tok.col)

    def _parse_magsq(self) -> MagSqStmt:
        tok = self.advance()  # 'magsq'
        self.expect('LPAREN')
        src = self.expect_id()
        self.expect('RPAREN')
        self.expect('KW_AS')
        name = self.expect_id()
        self.expect('SEMI')
        return MagSqStmt(src=src, name=name, line=tok.line, col=tok.col)

    def _parse_measure(self) -> MeasureStmt:
        tok = self.advance()  # 'measure'
        self.expect('LPAREN')
        src = self.expect_id()
        self.expect('COMMA')
        mode_tok = self.expect('MODE')
        self.expect('RPAREN')
        self.expect('KW_AS')
        name = self.expect_id()
        self.expect('SEMI')
        return MeasureStmt(
            src=src, mode=mode_tok.value, name=name,
            line=tok.line, col=tok.col,
        )

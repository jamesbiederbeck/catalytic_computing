"""Elaborator for Cook-Mertz v2.

NEW in v2: resolves field parameters and computes p, q, m for each scope.
v1 had no concept of field context — computation ring was implicit.

The elaborator:
  1. Reads the field(p, q) declaration (or infers defaults)
  2. Validates p is prime, q >= p, q is prime
  3. Attaches the resolved IRField to the AST for lowering
  4. Validates roots(p) matches the program-level field declaration
"""

from __future__ import annotations

from .ast_nodes import (
    Program, FieldDecl, CFieldDecl, EmbedStmt, RotateStmt, ComposeStmt,
    RootsStmt, CycloPhiStmt, MatPowStmt, CatalyticStmt, InterruptStmt,
    MeasureStmt, AddStmt, CEmbedStmt, ConjStmt, MagSqStmt, MulStmt, ASTNode,
)
from .field import IRField, compute_working_field, find_nonresidue

import sympy


class ElaborationError(Exception):
    pass


# Spec revision each construct was introduced in (§4 grammar history).
# Kept separate from package semver (see cmtz/__init__.py): v1->v2 was the
# only DSL-breaking transition, v2->v3 and v3->v4 were purely additive, so
# this registry is just "earliest spec label that accepts this construct" —
# it does not imply a major/minor split on its own.
CONSTRUCT_MIN_VERSION: dict[type, int] = {
    FieldDecl: 2,
    EmbedStmt: 2,
    RotateStmt: 2,
    ComposeStmt: 2,
    RootsStmt: 2,
    CycloPhiStmt: 2,
    MatPowStmt: 2,
    CatalyticStmt: 2,
    MeasureStmt: 2,
    InterruptStmt: 3,
    CFieldDecl: 4,
    AddStmt: 4,
    CEmbedStmt: 4,
    ConjStmt: 4,
    MagSqStmt: 4,
    MulStmt: 5,
}

MIN_SUPPORTED_VERSION = 2  # v1's cyclo()/cyclo_poly() lexicon no longer parses
MAX_SUPPORTED_VERSION = 5  # v5: IRMul


class Elaborator:
    """Resolve field parameters and validate algebraic consistency."""

    def __init__(self):
        self.ir_field: IRField | None = None
        self.version: int = MAX_SUPPORTED_VERSION

    def elaborate(self, program: Program) -> Program:
        """Elaborate a parsed AST: resolve fields, validate constraints."""
        self._resolve_version(program)
        self._resolve_field(program.field_decl)
        self._validate_statements(program.statements)
        return program

    def _resolve_version(self, program: Program) -> None:
        """Resolve the optional version(n) pragma and check every construct
        in the program against it. Absent a pragma, any supported version's
        constructs are accepted (current, pre-pragma behavior)."""
        decl = program.version_decl
        if decl is None:
            self.version = MAX_SUPPORTED_VERSION
            return

        if not (MIN_SUPPORTED_VERSION <= decl.n <= MAX_SUPPORTED_VERSION):
            raise ElaborationError(
                f"version({decl.n}): unsupported — this compiler supports "
                f"spec revisions {MIN_SUPPORTED_VERSION}..{MAX_SUPPORTED_VERSION} "
                f"(line {decl.line})"
            )
        self.version = decl.n

        if program.field_decl is not None:
            self._check_construct_version(program.field_decl)
        self._check_statements_version(program.statements)

    def _check_statements_version(self, stmts: list[ASTNode]) -> None:
        for stmt in stmts:
            self._check_construct_version(stmt)
            if isinstance(stmt, (CatalyticStmt, InterruptStmt)):
                self._check_statements_version(stmt.body)

    def _check_construct_version(self, node: ASTNode) -> None:
        min_version = CONSTRUCT_MIN_VERSION.get(type(node))
        if min_version is not None and min_version > self.version:
            raise ElaborationError(
                f"{type(node).__name__} requires version >= {min_version}, "
                f"but program declares version({self.version}) "
                f"(line {node.line})"
            )

    def _resolve_field(self, decl: FieldDecl | CFieldDecl | None) -> None:
        """Resolve field(p, q) or cfield(p, q) declaration into an IRField."""
        if decl is None:
            # No field declaration — defer to first roots() or matpow()
            return

        p = decl.p

        if not sympy.isprime(p):
            raise ElaborationError(
                f"field declaration: p={p} is not prime "
                f"(line {decl.line})"
            )

        if isinstance(decl, CFieldDecl):
            # Complex field: auto-compute non-residue c; q must be >= p
            c = find_nonresidue(p)
            if decl.q is not None:
                q = decl.q
                if not sympy.isprime(q):
                    raise ElaborationError(
                        f"cfield declaration: q={q} is not prime "
                        f"(line {decl.line})"
                    )
                if q < p:
                    raise ElaborationError(
                        f"cfield declaration: q={q} must be >= p={p} "
                        f"(line {decl.line})"
                    )
            else:
                # Auto-compute q: smallest prime >= p² (for F_{p²} multiplications)
                q = int(sympy.nextprime(p * p - 1))
            self.ir_field = IRField(p=p, q=q, c=c)
        else:
            q = decl.q if decl.q is not None else p
            if not sympy.isprime(q):
                raise ElaborationError(
                    f"field declaration: q={q} is not prime "
                    f"(line {decl.line})"
                )
            if q < p:
                raise ElaborationError(
                    f"field declaration: q={q} must be >= p={p} "
                    f"(line {decl.line})"
                )
            self.ir_field = IRField(p=p, q=q)

    def _validate_statements(self, stmts: list[ASTNode]) -> None:
        """Walk statements and validate field consistency."""
        for stmt in stmts:
            if isinstance(stmt, RootsStmt):
                self._validate_roots(stmt)
            elif isinstance(stmt, MatPowStmt):
                self._validate_matpow(stmt)
            elif isinstance(stmt, (CatalyticStmt, InterruptStmt)):
                self._validate_statements(stmt.body)
            elif isinstance(stmt, CEmbedStmt):
                self._validate_cembed(stmt)

    def _validate_roots(self, stmt: RootsStmt) -> None:
        """Validate roots(p) against program field."""
        if not sympy.isprime(stmt.p):
            raise ElaborationError(
                f"roots({stmt.p}): argument must be prime "
                f"(line {stmt.line})"
            )
        if self.ir_field is not None and stmt.p != self.ir_field.p:
            raise ElaborationError(
                f"roots({stmt.p}): p does not match field declaration "
                f"p={self.ir_field.p} (line {stmt.line})"
            )
        # If no field declaration yet, infer from roots()
        if self.ir_field is None:
            self.ir_field = IRField(p=stmt.p, q=stmt.p)

    def _validate_cembed(self, stmt: CEmbedStmt) -> None:
        """cembed requires a complex (cfield) declaration."""
        if self.ir_field is None or not self.ir_field.is_complex:
            raise ElaborationError(
                f"cembed requires a cfield declaration "
                f"(line {stmt.line})"
            )

    def _validate_matpow(self, stmt: MatPowStmt) -> None:
        """Validate matpow parameters and check field adequacy."""
        if stmt.d < 1:
            raise ElaborationError(
                f"matpow: degree d={stmt.d} must be >= 1 "
                f"(line {stmt.line})"
            )
        if stmt.epsilon <= 0 or stmt.epsilon > 1:
            raise ElaborationError(
                f"matpow: epsilon={stmt.epsilon} must be in (0, 1] "
                f"(line {stmt.line})"
            )
        if self.ir_field is None:
            raise ElaborationError(
                f"matpow requires a field declaration before use "
                f"(line {stmt.line})"
            )

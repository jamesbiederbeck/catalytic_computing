"""Compiler driver for Cook-Mertz v2.

Orchestrates the full pipeline:
  DSL source → Lex → Parse → Elaborate → Lower → Analyze → Optimize → Backend

Usage:
    from cmtz import compile_program
    result = compile_program(source, backend='python')
    result = compile_program(source, backend='analog')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .lexer import tokenize
from .parser import Parser
from .elaborator import Elaborator
from .lowering import lower
from .ir_nodes import IRProgram, IRRegion, IRCatalyticRegion
from .analysis import (
    check_cycles,
    check_field_consistency,
    verify_catalytic,
    propagate_costs,
    check_matpow_budget,
)
from .optimization import (
    fuse_rotations,
    eliminate_common_subexpressions,
    fold_constants,
)


@dataclass
class CompilationResult:
    """Result of compiling a Cook-Mertz DSL program."""
    ir_program: IRProgram
    backend_output: Any = None
    analysis_report: dict = field(default_factory=dict)
    optimization_stats: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def compile_program(
    source: str,
    backend: str = 'python',
    optimize: bool = True,
    check_budget: bool = True,
    device: str = 'cuda',
) -> CompilationResult:
    """Compile a Cook-Mertz DSL v2 source string.

    Args:
        source: DSL source code
        backend: 'python', 'torch', or 'analog'
        optimize: run optimization passes
        check_budget: verify Theorem 1.2 bounds
        device: torch device ('cuda' or 'cpu')

    Returns:
        CompilationResult with IR program and backend output
    """
    result = CompilationResult(ir_program=IRProgram())

    # ── Phase 1: Frontend ────────────────────────────────────────────

    try:
        tokens = tokenize(source)
    except Exception as e:
        result.errors.append(f"Lex error: {e}")
        return result

    try:
        parser = Parser(tokens)
        ast = parser.parse()
    except Exception as e:
        result.errors.append(f"Parse error: {e}")
        return result

    # ── Phase 2: Elaboration ─────────────────────────────────────────

    try:
        elaborator = Elaborator()
        ast = elaborator.elaborate(ast)
    except Exception as e:
        result.errors.append(f"Elaboration error: {e}")
        return result

    # ── Phase 3: IR Lowering ─────────────────────────────────────────

    try:
        ir_prog = lower(ast, ir_field=elaborator.ir_field)
        result.ir_program = ir_prog
    except Exception as e:
        result.errors.append(f"Lowering error: {e}")
        return result

    # ── Phase 4: Static Analysis ─────────────────────────────────────

    try:
        check_cycles(ir_prog)
        result.analysis_report['acyclic'] = True
    except Exception as e:
        result.errors.append(f"Cycle check: {e}")
        result.analysis_report['acyclic'] = False

    try:
        check_field_consistency(ir_prog)
        result.analysis_report['field_consistent'] = True
    except Exception as e:
        result.errors.append(f"Field consistency: {e}")
        result.analysis_report['field_consistent'] = False

    # Count advisory regions (IRRegion / catalytic {}) — not verified
    advisory = [n for _, n in ir_prog.nodes() if isinstance(n, IRRegion)]
    result.analysis_report['catalytic_advisory_count'] = len(advisory)

    try:
        deltas = verify_catalytic(ir_prog)
        result.analysis_report['catalytic_verified'] = True
        result.analysis_report['catalytic_deltas'] = [
            {'register': d.register, 'delta': d.delta_expr}
            for d in deltas
        ]
    except Exception as e:
        result.errors.append(f"Catalytic verification: {e}")
        result.analysis_report['catalytic_verified'] = False

    try:
        propagate_costs(ir_prog)
        result.analysis_report['costs_propagated'] = True
    except Exception as e:
        result.errors.append(f"Cost propagation: {e}")
        result.analysis_report['costs_propagated'] = False

    if check_budget:
        try:
            budget = check_matpow_budget(ir_prog)
            result.analysis_report['budget_check'] = budget
        except Exception as e:
            result.errors.append(f"Budget check: {e}")

    # ── Phase 5: Optimization ────────────────────────────────────────

    if optimize:
        stats = {}
        stats['rotations_fused'] = fuse_rotations(ir_prog)
        stats['cse_eliminated'] = eliminate_common_subexpressions(ir_prog)
        stats['constants_folded'] = fold_constants(ir_prog)
        result.optimization_stats = stats

    # ── Phase 6: Backend ─────────────────────────────────────────────

    try:
        if backend == 'python':
            from .backends.python_ref import PythonBackend
            if ir_prog.ir_field is not None:
                be = PythonBackend(ir_prog.ir_field)
                result.backend_output = be.eval_program(ir_prog)
            else:
                result.backend_output = {}

        elif backend == 'torch':
            from .backends.torch_backend import TorchBackend, TORCH_AVAILABLE
            if not TORCH_AVAILABLE:
                result.errors.append("PyTorch not available")
            elif ir_prog.ir_field is not None:
                be = TorchBackend(ir_prog.ir_field, device=device)
                result.backend_output = {'backend': 'torch', 'ready': True}
            else:
                result.errors.append("No field for torch backend")

        elif backend == 'analog':
            from .backends.analog_descriptor import emit_analog
            result.backend_output = dict(emit_analog(ir_prog))

        elif backend == 'glsl':
            from .backends.glsl_backend import emit_glsl
            bundle = emit_glsl(ir_prog)
            result.backend_output = {
                'glsl_source': bundle.glsl_source,
                'root_table': bundle.root_table,
                'buffers': [
                    {'binding': b.binding, 'name': b.name,
                     'type': b.element_type, 'count': b.element_count,
                     'usage': b.usage}
                    for b in bundle.buffers
                ],
                'workgroup_size': list(bundle.workgroup_size),
                'spirv_command': bundle.spirv_compile_command(),
            }

        elif backend == 'wasm':
            from .backends.wasm_backend import emit_wasm
            bundle = emit_wasm(ir_prog)
            result.backend_output = {
                'wat_source': bundle.wat_source,
                'root_table': bundle.root_table,
                'layout': {
                    'root_table_offset': bundle.layout.root_table_offset,
                    'root_table_bytes': bundle.layout.root_table_bytes,
                    'output_offset': bundle.layout.output_offset,
                    'measure_count': bundle.layout.measure_count,
                    'total_bytes': bundle.layout.total_bytes,
                },
                'assemble_command': bundle.assemble_command(),
            }

        else:
            result.errors.append(f"Unknown backend: {backend}")

    except Exception as e:
        result.errors.append(f"Backend error: {e}")

    return result

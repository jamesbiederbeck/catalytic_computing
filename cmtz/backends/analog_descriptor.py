"""Analog descriptor backend for Cook-Mertz v2.

NEW in v2: emits a JSON descriptor mapping compiled programs to
analog hardware parameters (magnonic, photonic, etc.).

The descriptor contains everything a hardware controller needs:
  - Field parameters (p, q)
  - Phase offsets θ_j = 2π·j/m for each root of unity
  - GEMM pass count (= recursive_calls from Lemma 2.1)
  - Register count (analog memory depth)
  - Restoration protocol hint
  - Tile size for systolic array dispatch
"""

from __future__ import annotations

import math
from typing import TypedDict

from ..ir_nodes import IRProgram, IRMatPow, IRNode
from ..field import IRField


class AnalogDescriptor(TypedDict):
    """JSON-serializable descriptor for analog hardware targeting."""
    field_p: int
    field_q: int
    root_order: int
    phase_offsets: list[float]
    gemm_passes: int
    register_count: int
    restoration_protocol: str
    tile_size: int
    total_basic_instructions: int


def emit_analog(prog: IRProgram) -> AnalogDescriptor:
    """Emit an analog hardware descriptor for a compiled program.

    Finds the IRMatPow node (if any) and uses its cost bounds
    to parameterize the hardware descriptor.
    """
    matpow = _find_matpow(prog)
    ir_field = prog.ir_field

    if ir_field is None:
        raise ValueError("Program has no field annotation — cannot emit analog descriptor")

    m = ir_field.m
    phase_offsets = [2 * math.pi * j / m for j in range(m)]

    if matpow is not None:
        costs = matpow.cost_bound()
        return AnalogDescriptor(
            field_p=ir_field.p,
            field_q=ir_field.q,
            root_order=m,
            phase_offsets=phase_offsets,
            gemm_passes=math.ceil(costs['recursive_calls']),
            register_count=math.ceil(costs['registers_Fq']),
            restoration_protocol='interferometric',
            tile_size=matpow.n_dim if matpow.n_dim > 0 else 1,
            total_basic_instructions=math.ceil(costs['basic_instructions']),
        )
    else:
        # No matpow — emit descriptor from aggregate program costs
        total_instructions = sum(
            n.basic_instructions for _, n in prog.nodes()
        )
        total_registers = max(
            (n.num_registers for _, n in prog.nodes()), default=1
        )
        return AnalogDescriptor(
            field_p=ir_field.p,
            field_q=ir_field.q,
            root_order=m,
            phase_offsets=phase_offsets,
            gemm_passes=1,
            register_count=total_registers,
            restoration_protocol='digital_reference',
            tile_size=1,
            total_basic_instructions=total_instructions,
        )


def _find_matpow(prog: IRProgram) -> IRMatPow | None:
    """Find the first IRMatPow node in the program."""
    for name, node in prog.nodes():
        if isinstance(node, IRMatPow):
            return node
    return None

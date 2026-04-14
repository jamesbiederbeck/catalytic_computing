#!/usr/bin/env python3
"""Compile tree_eval_depth3.cmtz and print the full compilation report."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cmtz import compile_program

SOURCE = Path(__file__).parent / "tree_eval_depth3.cmtz"

def main():
    source = SOURCE.read_text()
    print("=" * 60)
    print("Cook-Mertz v2 Compiler")
    print(f"Source: {SOURCE.name}")
    print("=" * 60)

    result = compile_program(source, backend='python', optimize=True)

    print(f"\nStatus: {'OK' if result.ok else 'ERRORS'}")

    if result.errors:
        print("\nErrors:")
        for e in result.errors:
            print(f"  ✗ {e}")

    print(f"\nIR Program ({len(result.ir_program)} nodes):")
    for name, node in result.ir_program.nodes():
        field_tag = f"F_{node.ir_field.p}" if node.ir_field else "no-field"
        cost = f"t={node.recursive_calls} s={node.basic_instructions} r={node.num_registers}"
        print(f"  {name:30s}  {type(node).__name__:22s}  [{field_tag}]  {cost}")

    print(f"\nAnalysis Report:")
    for k, v in result.analysis_report.items():
        if k == 'catalytic_deltas':
            print(f"  catalytic_deltas:")
            for d in v:
                print(f"    register={d['register']}  delta={d['delta']}")
        else:
            print(f"  {k}: {v}")

    print(f"\nOptimization Stats:")
    for k, v in result.optimization_stats.items():
        print(f"  {k}: {v}")

    if result.backend_output:
        print(f"\nBackend Output (python/F_p values):")
        for k, v in sorted(result.backend_output.items()):
            print(f"  {k} = {v}")

if __name__ == "__main__":
    main()

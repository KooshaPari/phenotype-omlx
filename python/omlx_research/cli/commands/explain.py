"""``explain`` — print the canonical contract + kernel candidates for an op.

With ``--shape`` we also pick a canonical layout of inputs/outputs and rank
the kernel-registry candidates that would be selected for that shape.

Without ``--shape`` we just print the contract prose.

Exit codes:
    0 — known op, output printed
    4 — op-kind is not in the recognized registry
"""

from __future__ import annotations

import argparse
import sys

from ._shared import OPERATOR_CONTRACTS, OPERATOR_KINDS, parse_shape


def _supported_set() -> str:
    return ", ".join(OPERATOR_KINDS)


def _kernel_selection(op_kind: str, shape: list[int]) -> list[str]:
    """Return the ordered list of kernel candidates a registry might pick.

    The first candidate is the *primary* selection; the rest are fallbacks.
    This is a deterministic synthetic ranking for research purposes, not a
    real auto-tuner.
    """
    contract = OPERATOR_CONTRACTS[op_kind]
    kernels = list(contract.get("kernels", ()))  # type: ignore[arg-type]
    # If a shape is provided, bias the primary toward the "metal" variant
    # when batch/seq dimensions look large. Otherwise preserve the registry
    # order from the contract.
    if shape and any(d >= 64 for d in shape):
        primary = next((k for k in kernels if "metal" in k), None)
    else:
        primary = next((k for k in kernels if "mlx" in k), None)
    if primary and primary in kernels:
        ordered = [primary] + [k for k in kernels if k != primary]
    else:
        ordered = kernels
    return ordered


def _layout_summary(op_kind: str, shape: list[int]) -> str:
    """Render a human-readable layout of inputs/outputs for the given shape."""
    contract = OPERATOR_CONTRACTS[op_kind]
    inputs = contract.get("inputs", [])  # type: ignore[assignment]
    outputs = contract.get("outputs", [])  # type: ignore[assignment]
    shape_str = ",".join(str(d) for d in shape)
    lines: list[str] = []
    for name, dtype in inputs:  # type: ignore[misc]
        lines.append(f"  in  {name:8s} dtype={dtype:4s} shape=({shape_str})")
    for name, dtype in outputs:  # type: ignore[misc]
        lines.append(f"  out {name:8s} dtype={dtype:4s} shape=({shape_str})")
    return "\n".join(lines)


def cmd_explain(args: argparse.Namespace) -> int:
    """CLI entry point: ``explain <op-kind> [--shape m,n,k,...]``"""
    op_kind = args.op_kind
    if op_kind not in OPERATOR_CONTRACTS:
        print(
            f"error: unknown op-kind {op_kind!r}. "
            f"Supported: {_supported_set()}",
            file=sys.stderr,
        )
        return 4

    contract = OPERATOR_CONTRACTS[op_kind]
    print(f"op_kind: {op_kind}")
    print(f"shape_rule: {contract['shape_rule']}")
    print(f"deps: {contract['deps']}")
    print("inputs:")
    for name, dtype in contract.get("inputs", []):  # type: ignore[misc]
        print(f"  - {name} ({dtype})")
    print("outputs:")
    for name, dtype in contract.get("outputs", []):  # type: ignore[misc]
        print(f"  - {name} ({dtype})")

    try:
        shape = parse_shape(args.shape)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if shape is not None:
        print("layout:")
        print(_layout_summary(op_kind, shape))
        print("kernel_selection (ordered):")
        for k in _kernel_selection(op_kind, shape):
            print(f"  - {k}")

    return 0

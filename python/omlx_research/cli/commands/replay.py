"""``replay`` — load an execution trace and replay it in human-readable form.

The trace shape is documented in ``_shared.validate_trace``. The expected
fields are::

    {
      "plan_id":   str,
      "op_id":     str,
      "selected":  {"kernel": str, "reason": str, "fallback": str},
      "rejected":  [{"kernel": str, "reason": str}, ...],
      "latency_ns": float | None   # optional
    }

Exit codes:
    0 — success
    5 — trace file missing
    6 — trace file is malformed (parse or schema)
"""

from __future__ import annotations

import argparse
import json
import sys

from ._shared import validate_trace


def _load_trace(path: str) -> tuple[dict | None, int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"error: trace file not found: {path}", file=sys.stderr)
        return None, 5
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in {path}: {e}", file=sys.stderr)
        return None, 6
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return None, 5

    errors = validate_trace(data)
    if errors:
        print(f"error: trace {path} failed schema validation:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return None, 6
    return data, 0


def _print_rejected(trace: dict) -> None:
    print("rejected candidates (in order):")
    for i, r in enumerate(trace.get("rejected") or []):
        if not isinstance(r, dict):
            print(f"  {i + 1}. <malformed entry: {r!r}>")
            continue
        kernel = r.get("kernel", "?")
        reason = r.get("reason", "?")
        print(f"  {i + 1}. {kernel}  — {reason}")


def _print_selected(trace: dict) -> None:
    sel = trace.get("selected") or {}
    print("selected candidate:")
    print(f"  kernel  : {sel.get('kernel', '?')}")
    print(f"  reason  : {sel.get('reason', '?')}")
    print(f"  fallback: {sel.get('fallback', '?')}")
    lat = trace.get("latency_ns")
    if lat is not None:
        print(f"  latency : {lat} ns")


def cmd_replay(args: argparse.Namespace) -> int:
    """CLI entry point: ``replay <trace-file> [--filter-rejected|--filter-selected]``"""
    trace, rc = _load_trace(args.trace_file)
    if trace is None:
        return rc

    print(f"plan_id : {trace.get('plan_id', '?')}")
    print(f"op_id   : {trace.get('op_id', '?')}")

    show_rejected = not args.filter_rejected
    show_selected = not args.filter_selected

    if show_rejected:
        _print_rejected(trace)
    if show_selected:
        _print_selected(trace)

    if not show_rejected and not show_selected:
        print("(nothing to print — both filters active)")

    return 0

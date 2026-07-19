"""``compare`` — side-by-side comparison of two execution traces.

Output is JSON to stdout. The comparison covers:
- plan_id (must match, else exit 7)
- op_id (must match)
- selected kernel, fallback used
- top-of-list rejected reasons (first up to N)
- latency p95 if present

Exit codes:
    0 — comparison produced
    2 — malformed JSON in either trace
    5 — trace file missing
    7 — plan_ids differ (the user almost certainly passed the wrong pair)
"""

from __future__ import annotations

import argparse
import json
import sys

from ._shared import validate_trace


TOP_REJECTED = 3  # how many rejected reasons to surface


def _load(path: str, label: str) -> tuple[dict | None, int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"error: {label} trace not found: {path}", file=sys.stderr)
        return None, 5
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in {label} trace {path}: {e}", file=sys.stderr)
        return None, 2
    except OSError as e:
        print(f"error: cannot read {label} trace {path}: {e}", file=sys.stderr)
        return None, 5
    errors = validate_trace(data)
    if errors:
        print(f"error: {label} trace {path} failed schema:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return None, 2
    return data, 0


def _summary(trace: dict) -> dict:
    sel = trace.get("selected") or {}
    rejected = trace.get("rejected") or []
    top_reasons = [
        r.get("reason", "?") for r in rejected[:TOP_REJECTED] if isinstance(r, dict)
    ]
    return {
        "plan_id": trace.get("plan_id"),
        "op_id": trace.get("op_id"),
        "selected_kernel": sel.get("kernel"),
        "fallback_used": sel.get("fallback"),
        "top_rejected_reasons": top_reasons,
        "latency_p95_ns": trace.get("latency_ns"),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    """CLI entry point: ``compare <trace-a> <trace-b>``"""
    a, rc = _load(args.trace_a, "a")
    if a is None:
        return rc
    b, rc = _load(args.trace_b, "b")
    if b is None:
        return rc

    if a.get("plan_id") != b.get("plan_id"):
        print(
            f"error: plan_id mismatch: {a.get('plan_id')!r} vs {b.get('plan_id')!r}",
            file=sys.stderr,
        )
        return 7

    out = {
        "plan_id": a.get("plan_id"),
        "op_id_a": a.get("op_id"),
        "op_id_b": b.get("op_id"),
        "op_id_match": a.get("op_id") == b.get("op_id"),
        "a": _summary(a),
        "b": _summary(b),
    }
    sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0

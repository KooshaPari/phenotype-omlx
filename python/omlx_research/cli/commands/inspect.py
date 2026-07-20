"""``inspect`` — load + validate a model-plan JSON file and print a summary.

This is a *research* CLI, so the schema is enforced by hand-rolled checks
(see ``_shared.validate_plan``) rather than the ``jsonschema`` package — we
want zero new top-level dependencies.

Exit codes:
    0 — plan loaded, validated, printed
    2 — file missing or JSON parse error
    3 — schema validation failed
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ._shared import validate_plan


# ---------------------------------------------------------------------------
# Synthetic plan for --empty
# ---------------------------------------------------------------------------

def _synthetic_plan() -> dict[str, Any]:
    """Return a tiny, known-good 2-operator plan for the --empty flag."""
    return {
        "plan_id": "synthetic-2op",
        "name": "synthetic-mini",
        "family": "research-toy",
        "scheduler_policy": "fifo",
        "operators": [
            {
                "op_id": "op0",
                "kind": "DenseMatmul",
                "inputs": ["A", "B"],
                "outputs": ["Y0"],
            },
            {
                "op_id": "op1",
                "kind": "RMSNorm",
                "inputs": ["Y0", "weight"],
                "outputs": ["Y1"],
            },
        ],
        "states": [
            {
                "state_id": "st0",
                "kind": "tensor",
                "persistence": "scratch",
                "dtype": "f16",
                "owning_op": "op0",
            },
        ],
        "edges": [
            {"from_id": "op0", "to_id": "op1"},
        ],
    }


# ---------------------------------------------------------------------------
# Plan loader
# ---------------------------------------------------------------------------

def _load_plan(path: str | None, empty: bool) -> tuple[dict[str, Any] | None, int]:
    """Load the plan JSON. Returns (plan, exit_code).

    On error prints to stderr and returns ``(None, exit_code)``.
    """
    if empty:
        return _synthetic_plan(), 0
    if path is None:
        print(
            "error: must provide a plan file path or pass --empty",
            file=sys.stderr,
        )
        return None, 2
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"error: plan file not found: {path}", file=sys.stderr)
        return None, 2
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in {path}: {e}", file=sys.stderr)
        return None, 2
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return None, 2
    if not isinstance(data, dict):
        print("error: plan must be a JSON object at the top level", file=sys.stderr)
        return None, 2
    return data, 0


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _print_summary(plan: dict[str, Any], errors: list[str]) -> None:
    print(f"plan_id           : {plan.get('plan_id', '<missing>')}")
    print(f"name              : {plan.get('name', '<missing>')}")
    print(f"family            : {plan.get('family', '<missing>')}")
    print(f"scheduler_policy  : {plan.get('scheduler_policy', '<missing>')}")
    ops = plan.get("operators") or []
    states = plan.get("states") or []
    edges = plan.get("edges") or []
    print(f"operator_count    : {len(ops)}")
    print(f"state_count       : {len(states)}")
    print(f"edge_count        : {len(edges)}")
    if errors:
        print(f"validation        : FAIL ({len(errors)} error(s))")
        for e in errors:
            print(f"  - {e}")
    else:
        print("validation        : OK")


def _print_states(plan: dict[str, Any]) -> None:
    states = plan.get("states") or []
    if not states:
        print("(no states in plan)")
        return
    print("states:")
    for st in states:
        print(
            f"  - {st.get('state_id', '?')} kind={st.get('kind', '?')} "
            f"persistence={st.get('persistence', '?')} "
            f"dtype={st.get('dtype', '?')} owning_op={st.get('owning_op', '?')}"
        )


def _print_edges(plan: dict[str, Any]) -> None:
    edges = plan.get("edges") or []
    if not edges:
        print("(no edges in plan)")
        return
    print("edges:")
    for e in edges:
        print(f"  - {e.get('from_id', '?')} -> {e.get('to_id', '?')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    """CLI entry point: ``inspect <model-plan.json> [--show-states] [--show-deps]``"""
    plan, rc = _load_plan(args.plan, bool(args.empty))
    if plan is None:
        return rc

    errors = validate_plan(plan)
    _print_summary(plan, errors)

    if args.show_states:
        _print_states(plan)
    if args.show_deps:
        _print_edges(plan)

    return 3 if errors else 0


# ---------------------------------------------------------------------------
# Negative example (used in tests + documented in docstring)
# ---------------------------------------------------------------------------
#
# A known-bad plan that should fail validation:
#
#   {
#     "plan_id": "bad",
#     "name": "bad-plan",
#     "family": "x",
#     "scheduler_policy": "round-robin",   # not in SCHEDULER_POLICIES
#     "operators": [
#       {"op_id": "op0", "kind": "BizarreOp"}   # kind unknown
#     ]
#   }
#
# Reason for failure: scheduler_policy not in {fifo,priority,critical_path,
# dataflow}, and operators[0].kind is not in the OPERATOR_KINDS registry.

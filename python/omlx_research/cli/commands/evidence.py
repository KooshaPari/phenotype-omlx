"""``evidence`` — generate an evidence bundle for a model plan.

The bundle is deterministic, fictional-but-stable and contains:
- plan summary
- validation status
- a fictional kernel trace (deterministic hash of plan_id)
- a fictional tuning record (deterministic hash of plan_id)
- the command line that produced the bundle
- sys-info (platform, python version)
- git rev (best-effort; falls back to ``"unknown"``)

We write the bundle to stdout AND to ``./omlx-evidence-<unix-ts>.json``.

Exit codes:
    0 — bundle produced
    8 — plan file missing or unreadable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

from ._shared import validate_plan


# Repo root for `git rev-parse`. We try a couple of likely locations before
# giving up — the absolute path baked into the project tree is the one that
# matches the test environment.
_REPO_ROOTS: tuple[str, ...] = (
    "/Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-registry/registry/absorbed-crates/phenotype-omlx",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    ),
)


def _git_rev() -> str:
    for root in _REPO_ROOTS:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            s = out.decode("utf-8", errors="replace").strip()
            if s:
                return s
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    return "unknown"


def _stable_hash(seed: str, length: int = 16) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:length]


def _fictional_trace(plan: dict) -> dict:
    """Deterministic-but-fictional kernel trace keyed off plan_id."""
    seed = str(plan.get("plan_id", "unknown"))
    return {
        "trace_id": _stable_hash("trace|" + seed),
        "events": [
            {"step": 1, "op_id": "op0", "kernel": "op0_metal",   "ok": True},
            {"step": 2, "op_id": "op1", "kernel": "op1_mlx",     "ok": True},
            {"step": 3, "op_id": "op2", "kernel": "op2_fallback","ok": True},
        ],
    }


def _fictional_tuning(plan: dict) -> dict:
    seed = str(plan.get("plan_id", "unknown"))
    return {
        "tuning_id": _stable_hash("tune|" + seed),
        "selected_kernel": "op0_metal",
        "samples": 16,
        "p50_ns": 1234,
        "p95_ns": 1456,
    }


def _sys_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "node": platform.node(),
    }


def _load_plan(path: str) -> tuple[dict | None, int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"error: plan file not found: {path}", file=sys.stderr)
        return None, 8
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in {path}: {e}", file=sys.stderr)
        return None, 8
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return None, 8
    if not isinstance(data, dict):
        print("error: plan must be a JSON object at the top level", file=sys.stderr)
        return None, 8
    return data, 0


def _build_bundle(plan: dict, argv: list[str]) -> dict[str, Any]:
    errors = validate_plan(plan)
    return {
        "schema_version": 1,
        "unix_ts": int(time.time()),
        "command": "evidence " + " ".join(argv),
        "plan": {
            "plan_id": plan.get("plan_id"),
            "name": plan.get("name"),
            "family": plan.get("family"),
            "scheduler_policy": plan.get("scheduler_policy"),
            "operator_count": len(plan.get("operators") or []),
            "state_count": len(plan.get("states") or []),
            "edge_count": len(plan.get("edges") or []),
        },
        "validation": {
            "ok": not errors,
            "errors": errors,
        },
        "kernel_trace": _fictional_trace(plan),
        "tuning_record": _fictional_tuning(plan),
        "sys_info": _sys_info(),
        "git_rev": _git_rev(),
    }


def cmd_evidence(args: argparse.Namespace) -> int:
    """CLI entry point: ``evidence <plan-file>``"""
    plan, rc = _load_plan(args.plan_file)
    if plan is None:
        return rc

    bundle = _build_bundle(plan, args._argv or [])
    payload = json.dumps(bundle, indent=2, sort_keys=True)

    # Write to a file in CWD; do this *before* stdout so tests can read it
    # even if the stdout buffer is line-buffered oddly.
    out_path = os.path.abspath(
        os.path.join(os.getcwd(), f"omlx-evidence-{bundle['unix_ts']}.json")
    )
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    except OSError as e:
        print(f"warning: could not write evidence file: {e}", file=sys.stderr)

    sys.stdout.write(payload + "\n")
    sys.stdout.flush()
    return 0

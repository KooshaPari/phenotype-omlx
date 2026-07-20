"""``gates`` — CRUD against per-kernel quality-gate configurations.

Storage layout (mirrors ``promote.py``)::

    .omlx/cache/gates/<kernel_id>.json
        {
          "schema_version": 1,
          "kernel_id": "<kernel_id>",
          "updated_at_unix_ms": <ms>,
          "gates": [
            {"id": "mmlu", "threshold": 0.85, "direction": "at_least", "note": "..."},
            ...
          ]
        }

Subcommands:
    list   <kernel_id>                    — print all gates for the kernel
    add    <kernel_id> --gate ID          — add a gate (--threshold, --at-least/--at-most, --note)
    remove <kernel_id> --gate ID          — remove a gate by id
    check  <kernel_id> --gate ID --score  — evaluate a single gate against a score

Exit codes:
    0 — operation succeeded
    2 — user error: missing args, unknown gate id, malformed threshold
    3 — internal error: cannot read or write gate file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


GATES_DIRNAME = "gates"
SCHEMA_VERSION = 1


def _project_root() -> str:
    """Re-derive the project root using the same heuristic as ``promote``."""
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def cache_root() -> str:
    """Return ``<project>/.omlx/cache``."""
    return os.path.join(_project_root(), ".omlx", "cache")


def gates_path(kernel_id: str) -> str:
    """Absolute path to the per-kernel gates config file."""
    return os.path.join(cache_root(), GATES_DIRNAME, f"{kernel_id}.json")


# ---------------------------------------------------------------------------
# Storage helpers.
# ---------------------------------------------------------------------------

def _empty_config(kernel_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kernel_id": kernel_id,
        "updated_at_unix_ms": int(time.time() * 1000),
        "gates": [],
    }


def _normalize_gate(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Coerce a gate dict into the canonical shape; return ``None`` if invalid."""
    if not isinstance(raw, dict):
        return None
    gid = raw.get("id")
    if not isinstance(gid, str) or not gid:
        return None
    threshold = raw.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return None
    direction = raw.get("direction", "at_least")
    if direction not in ("at_least", "at_most"):
        return None
    note = raw.get("note", "")
    if not isinstance(note, str):
        note = ""
    return {
        "id": gid,
        "threshold": float(threshold),
        "direction": direction,
        "note": note,
    }


def load_config(kernel_id: str) -> dict[str, Any]:
    """Read the gate config for ``kernel_id``; return an empty config if absent."""
    path = gates_path(kernel_id)
    if not os.path.exists(path):
        return _empty_config(kernel_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read gates file {path}: {e}", file=sys.stderr)
        # Fall back to empty so the CLI doesn't refuse subsequent writes.
        return _empty_config(kernel_id)
    if not isinstance(data, dict):
        return _empty_config(kernel_id)
    raw_gates = data.get("gates")
    gates: list[dict[str, Any]] = []
    if isinstance(raw_gates, list):
        for g in raw_gates:
            norm = _normalize_gate(g)
            if norm is not None:
                gates.append(norm)
    return {
        "schema_version": SCHEMA_VERSION,
        "kernel_id": kernel_id,
        "updated_at_unix_ms": data.get("updated_at_unix_ms", int(time.time() * 1000)),
        "gates": gates,
    }


def save_config(kernel_id: str, config: dict[str, Any]) -> str:
    """Persist the gate config; return the absolute path."""
    path = gates_path(kernel_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config["updated_at_unix_ms"] = int(time.time() * 1000)
    payload = json.dumps(config, indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload + "\n")
    return path


def find_gate(config: dict[str, Any], gate_id: str) -> dict[str, Any] | None:
    """Return the gate dict with matching id, or ``None``."""
    for g in config.get("gates", []):
        if g.get("id") == gate_id:
            return g
    return None


# ---------------------------------------------------------------------------
# Subcommand handlers.
# ---------------------------------------------------------------------------

def _parse_threshold(raw: str | float | int | None) -> float:
    if raw is None:
        raise ValueError("--threshold is required")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
    else:
        try:
            value = float(str(raw).strip())
        except (ValueError, AttributeError) as e:
            raise ValueError(f"threshold {raw!r} is not a number") from e
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("threshold must be finite")
    return value


def cmd_gates_list(args: argparse.Namespace) -> int:
    """``gates list <kernel_id> [--json]``"""
    config = load_config(args.kernel_id)
    if getattr(args, "json", False):
        sys.stdout.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": True,
                    "kernel_id": args.kernel_id,
                    "gates": config["gates"],
                    "cache_path": gates_path(args.kernel_id),
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
        return 0

    if not config["gates"]:
        print(f"no gates configured for kernel_id={args.kernel_id}")
        return 0
    print(f"kernel_id         : {args.kernel_id}")
    print(f"updated_at_unix_ms: {config['updated_at_unix_ms']}")
    print(f"gates ({len(config['gates'])}):")
    for g in config["gates"]:
        print(
            f"  - {g['id']:24s} threshold={g['threshold']:.4f} "
            f"direction={g['direction']}"
            + (f"  note={g['note']!r}" if g.get("note") else "")
        )
    return 0


def cmd_gates_add(args: argparse.Namespace) -> int:
    """``gates add <kernel_id> --gate ID [--threshold N] [--at-least|--at-most] [--note ...]``"""
    if not getattr(args, "gate", None):
        print("error: --gate ID is required for 'add'", file=sys.stderr)
        return 2
    try:
        threshold = _parse_threshold(getattr(args, "threshold", None))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    direction = "at_most" if getattr(args, "at_most", False) else "at_least"
    note = getattr(args, "note", "") or ""

    config = load_config(args.kernel_id)
    existing = find_gate(config, args.gate)
    new_gate = {
        "id": args.gate,
        "threshold": threshold,
        "direction": direction,
        "note": note,
    }
    if existing is not None:
        # Update in place rather than append a duplicate; keeps the file tidy.
        for i, g in enumerate(config["gates"]):
            if g.get("id") == args.gate:
                config["gates"][i] = new_gate
                break
        verb = "updated"
    else:
        config["gates"].append(new_gate)
        verb = "added"

    try:
        path = save_config(args.kernel_id, config)
    except OSError as e:
        print(f"error: could not write gates file: {e}", file=sys.stderr)
        return 3

    if getattr(args, "json", False):
        sys.stdout.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": True,
                    "verb": verb,
                    "kernel_id": args.kernel_id,
                    "gate": new_gate,
                    "cache_path": path,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
    else:
        print(f"{verb} gate {new_gate['id']!r} for {args.kernel_id}")
        print(f"  threshold : {new_gate['threshold']}")
        print(f"  direction : {new_gate['direction']}")
        print(f"  note      : {new_gate['note']!r}" if new_gate["note"] else "  note      : (none)")
        print(f"  cache_path: {path}")
    return 0


def cmd_gates_remove(args: argparse.Namespace) -> int:
    """``gates remove <kernel_id> --gate ID``"""
    if not getattr(args, "gate", None):
        print("error: --gate ID is required for 'remove'", file=sys.stderr)
        return 2

    config = load_config(args.kernel_id)
    before = len(config["gates"])
    config["gates"] = [g for g in config["gates"] if g.get("id") != args.gate]
    after = len(config["gates"])
    if after == before:
        print(f"error: no gate {args.gate!r} for {args.kernel_id}", file=sys.stderr)
        return 2

    try:
        path = save_config(args.kernel_id, config)
    except OSError as e:
        print(f"error: could not write gates file: {e}", file=sys.stderr)
        return 3

    if getattr(args, "json", False):
        sys.stdout.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": True,
                    "removed": args.gate,
                    "kernel_id": args.kernel_id,
                    "remaining": after,
                    "cache_path": path,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
    else:
        print(f"removed gate {args.gate!r} from {args.kernel_id} (remaining: {after})")
        print(f"  cache_path: {path}")
    return 0


def cmd_gates_check(args: argparse.Namespace) -> int:
    """``gates check <kernel_id> --gate ID --score FLOAT``"""
    if not getattr(args, "gate", None):
        print("error: --gate ID is required for 'check'", file=sys.stderr)
        return 2
    if getattr(args, "score", None) is None:
        print("error: --score FLOAT is required for 'check'", file=sys.stderr)
        return 2
    try:
        score = _parse_threshold(args.score)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    config = load_config(args.kernel_id)
    gate = find_gate(config, args.gate)
    if gate is None:
        print(
            f"error: no gate {args.gate!r} configured for {args.kernel_id}",
            file=sys.stderr,
        )
        return 2

    threshold = float(gate["threshold"])
    direction = gate["direction"]
    passes = score >= threshold if direction == "at_least" else score <= threshold

    payload = {
        "schema_version": 1,
        "kernel_id": args.kernel_id,
        "gate": args.gate,
        "score": score,
        "threshold": threshold,
        "direction": direction,
        "passes": passes,
        "observed_delta": (
            round(score - threshold, 6)
            if direction == "at_least"
            else round(threshold - score, 6)
        ),
    }
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sign = ">= " if direction == "at_least" else "<= "
        status = "OK" if passes else "FAIL"
        print(f"gate   : {args.gate}")
        print(f"score  : {score}")
        print(f"rule   : score {sign}{threshold}")
        print(f"result : {status}")
    return 0


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {
    "list": cmd_gates_list,
    "add": cmd_gates_add,
    "remove": cmd_gates_remove,
    "check": cmd_gates_check,
}


def cmd_gates(args: argparse.Namespace) -> int:
    """CLI entry point: ``gates <list|add|remove|check> <kernel_id> [...]``"""
    sub = getattr(args, "gates_action", None)
    fn = _SUBCOMMANDS.get(sub or "")
    if fn is None:
        print(
            f"error: unknown subcommand {sub!r}; expected one of "
            f"{sorted(_SUBCOMMANDS.keys())}",
            file=sys.stderr,
        )
        return 2
    return fn(args)

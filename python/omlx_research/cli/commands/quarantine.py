"""``quarantine`` — append a Hold or Rollback audit entry for a kernel.

Mirrors the ``PromotionAction::{Hold, Rollback}`` audit-summary variants
in ``perf-core/kernel-registry::quality``. A quarantine action is the
*counterpart* to ``promote``: instead of approving a candidate for
production it pushes the candidate onto a watch list and the entry is
appended to ``.omlx/cache/audit.jsonl`` (one JSON object per line).

If a cached ``PromotionRecord`` already exists for the kernel the
quarantine action reuses its evidence rows so the audit trail is
self-consistent. Otherwise an empty record is built (no evidence, no
gates) — the entry still records *what* happened and *why*.

Exit codes:
    0 — audit entry appended
    2 — user error: missing reason
    3 — internal error: cannot write audit file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from . import promote as _promote


AUDIT_FILENAME = "audit.jsonl"


def _project_root() -> str:
    """Re-derive the project root using the same heuristic as ``promote``."""
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def cache_root() -> str:
    """Return ``<project>/.omlx/cache`` — same layout as the Rust registry."""
    return os.path.join(_project_root(), ".omlx", "cache")


def audit_path() -> str:
    """Absolute path to the audit JSONL file."""
    return os.path.join(cache_root(), AUDIT_FILENAME)


def _human_summary(entry: dict[str, Any], cache_path: str) -> str:
    return "\n".join([
        f"action            : {entry['action']}",
        f"candidate_id      : {entry['candidate_id']}",
        f"reason            : {entry['reason']}",
        f"unix_ts           : {entry['unix_ts']}",
        f"approver          : {entry['approver']}",
        f"content_hash      : {entry['content_hash'][:16]}{'...' if entry['content_hash'] else ''}",
        f"audit_path        : {cache_path}",
        "quarantine        : OK",
    ])


def _json_summary(entry: dict[str, Any], cache_path: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": True,
        "audit_path": cache_path,
        "entry": entry,
    }


def _build_entry(
    kernel_id: str,
    reason: str,
    action: str,
    approver: str,
) -> dict[str, Any]:
    """Build the audit-trail entry (one JSON object per line).

    The shape is intentionally flat: ``action`` is one of
    ``hold`` / ``rollback`` (matching ``PromotionAction`` in the Rust
    crate), ``candidate_id`` is the kernel id, ``reason`` is the
    user-supplied string, and ``record`` is a minimal PromotionRecord
    carrying the kernel id, evidence (if cached), and a content hash.
    """
    cached = _promote._load_cached_record(kernel_id)  # type: ignore[attr-defined]
    gates = list(cached.get("gates", [])) if isinstance(cached, dict) else []
    evidence = list(cached.get("evidence", [])) if isinstance(cached, dict) else []
    src_rev = (
        cached.get("source_revision", "synthetic-rev-0")
        if isinstance(cached, dict) and isinstance(cached.get("source_revision"), str)
        else "synthetic-rev-0"
    )
    justification = (
        cached.get("justification", "")
        if isinstance(cached, dict) and isinstance(cached.get("justification"), str)
        else ""
    )

    record: dict[str, Any] = {
        "candidate_id": kernel_id,
        "source_revision": src_rev,
        "approved_at_unix_ms": int(time.time() * 1000),
        "approver": approver,
        "gates": gates,
        "evidence": evidence,
        "justification": justification,
        "tuning_record_id": None,
        "signature": None,
        "content_hash": "",
    }
    record["content_hash"] = _promote.content_hash(record)  # type: ignore[attr-defined]

    return {
        "schema_version": 1,
        "action": action,
        "candidate_id": kernel_id,
        "reason": reason,
        "unix_ts": int(time.time()),
        "approver": approver,
        "content_hash": record["content_hash"],
        "record": record,
    }


def _append_audit_line(entry: dict[str, Any]) -> str:
    """Append one JSON line to ``.omlx/cache/audit.jsonl``; return the path."""
    path = audit_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def cmd_quarantine(args: argparse.Namespace) -> int:
    """CLI entry point: ``quarantine <kernel_id> --reason <text> [--json]``"""
    reason = (args.reason or "").strip()
    if not reason:
        print("error: --reason must be a non-empty string", file=sys.stderr)
        return 2

    action = getattr(args, "action", None) or "hold"
    if action not in ("hold", "rollback"):
        print(f"error: unknown action {action!r} (expected hold|rollback)", file=sys.stderr)
        return 2

    approver = (
        getattr(args, "approver", None) or os.environ.get("USER") or "unknown"
    )
    entry = _build_entry(args.kernel_id, reason, action, approver)

    try:
        path = _append_audit_line(entry)
    except OSError as e:
        print(f"error: could not write audit file: {e}", file=sys.stderr)
        return 3

    if getattr(args, "json", False):
        sys.stdout.write(
            json.dumps(_json_summary(entry, path), indent=2, sort_keys=True) + "\n"
        )
    else:
        print(_human_summary(entry, path))
    return 0

"""``promote`` — validate a candidate against quality gates and sign a record.

Mirrors ``perf-core/kernel-registry::PromotionValidator::promote`` (Rust):

- each gate must have matching evidence with a passing score
- if a signing key is supplied, the record is HMAC-SHA256 signed
- the record's ``content_hash`` is always recomputed from the canonical
  fields (every field *except* ``signature`` and ``content_hash``)

Reads an existing ``PromotionRecord`` from
``.omlx/cache/promotion/<kernel_id>.json`` when one exists so a caller
can re-validate a previously-built candidate without rebuilding the
evidence rows. When no record exists an in-memory candidate is
constructed with synthetic (deterministic, hash-derived) evidence so
the workflow can be exercised end-to-end from the CLI without a real
benchmark harness.

On success:
- signed PromotionRecord is written to ``.omlx/cache/promotion/<kernel_id>.json``
- a human-readable summary is printed (or a JSON envelope with --json)

On failure:
- exit 2 with a structured error showing which gate failed and the
  observed/threshold values

Exit codes:
    0 — record promoted (signed and cached)
    2 — user error: malformed --gates, failed validation, bad key
    3 — internal error: cannot write cache file
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Cache layout — matches the convention in the Rust registry.
# ---------------------------------------------------------------------------

PROMOTION_DIRNAME = "promotion"


def _project_root() -> str:
    """Return the absolute path of the .omlx cache root.

    The Rust registry writes under ``<repo>/.omlx/cache/...`` and the CLI
    mirrors that. Tests override this via monkeypatch.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    # here == .../omlx_research/cli/commands/  -> walk up 4 levels
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def cache_root() -> str:
    """Return ``<project>/.omlx/cache`` — absolute."""
    return os.path.join(_project_root(), ".omlx", "cache")


def promotion_path(kernel_id: str) -> str:
    """Absolute path to the per-kernel PromotionRecord cache file."""
    return os.path.join(cache_root(), PROMOTION_DIRNAME, f"{kernel_id}.json")


# ---------------------------------------------------------------------------
# Canonical record model (mirrors perf-core/kernel-registry::quality).
# ---------------------------------------------------------------------------

#: Field order is part of the contract: the content hash is computed over
#: this key order so re-serialization must preserve it.
PROMOTION_RECORD_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "source_revision",
    "approved_at_unix_ms",
    "approver",
    "gates",
    "evidence",
    "justification",
    "tuning_record_id",
)
def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(record: dict[str, Any]) -> bytes:
    """Canonical JSON bytes for ``record`` (key-ordered, compact).

    Only fields in ``PROMOTION_RECORD_FIELDS`` are included. ``signature``
    and ``content_hash`` are deliberately excluded so the hash stays
    stable across re-signing. Inner dicts (gates, evidence) are
    serialized with ``sort_keys=True`` so the hash survives a round-trip
    through a file that was written with sorted keys.
    """
    intermediate: dict[str, Any] = {f: record.get(f) for f in PROMOTION_RECORD_FIELDS}
    return json.dumps(intermediate, separators=(",", ":"), sort_keys=True).encode("utf-8")


def content_hash(record: dict[str, Any]) -> str:
    """Stable SHA-256 hex of the canonical fields (no signature/content_hash)."""
    return _sha256_hex(canonical_bytes(record))


def sign_record(record: dict[str, Any], signing_key: bytes) -> str:
    """Compute HMAC-SHA256 hex over the canonical bytes.

    The Rust registry uses raw SHA-256 for its built-in signing; we use
    HMAC-SHA256 here because the task spec requires ``hmac`` and HMAC is
    the standard upgrade when a stable MAC is needed (see the docstring
    on ``PromotionRecord::sign_with`` in quality.rs).
    """
    return hmac.new(signing_key, canonical_bytes(record), hashlib.sha256).hexdigest()
# ---------------------------------------------------------------------------
# Gate parsing.
# ---------------------------------------------------------------------------

#: Map of CLI direction flag → GateDirection value (mirrors quality.rs).
_AT_LEAST = "at_least"
_AT_MOST = "at_most"


def parse_gates(spec: str) -> list[dict[str, Any]]:
    """Parse ``--gates mmlu=0.85,gpqa=0.75`` into gate dicts.

    Each gate defaults to ``AtLeast`` (higher is better). The threshold
    must be a finite float in the range [-1e9, 1e9]; anything else is a
    user error and exits 2.

    Returns a list of dicts shaped like::

        [{"id": "mmlu", "threshold": 0.85, "direction": "at_least", "note": ""}, ...]
    """
    if not spec:
        raise ValueError("empty --gates spec")
    out: list[dict[str, Any]] = []
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"gate entry {tok!r} must be 'id=threshold'")
        gid, thr = tok.split("=", 1)
        gid = gid.strip()
        if not gid:
            raise ValueError(f"empty gate id in {tok!r}")
        try:
            value = float(thr)
        except ValueError as e:
            raise ValueError(f"gate {gid!r}: threshold {thr!r} not a float") from e
        if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
            raise ValueError(f"gate {gid!r}: threshold must be finite")
        out.append({
            "id": gid,
            "threshold": value,
            "direction": _AT_LEAST,
            "note": "",
        })
    if not out:
        raise ValueError("--gates spec parsed to zero gates")
    return out


def gate_passes(gate: dict[str, Any], score: float) -> bool:
    """Mirror ``QualityGate::passes`` from quality.rs."""
    if gate["direction"] == _AT_LEAST:
        return score >= gate["threshold"]
    if gate["direction"] == _AT_MOST:
        return score <= gate["threshold"]
    raise ValueError(f"unknown gate direction {gate['direction']!r}")


# ---------------------------------------------------------------------------
# Candidate construction (when no cache record exists yet).
# ---------------------------------------------------------------------------

def _synthetic_evidence(kernel_id: str, gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build deterministic synthetic evidence for the in-memory candidate.

    Score is a stable function of ``(kernel_id, gate_id)`` so the CLI is
    deterministic across runs without a real benchmark harness. To pass a
    realistic gate, choose a threshold that falls below the score — the
    default score falls in ``[0.5, 1.0)``.
    """
    out: list[dict[str, Any]] = []
    for gate in gates:
        seed = f"{kernel_id}|{gate['id']}"
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        # Use the first 4 bytes as an unsigned int, map to [0.5, 1.0).
        n = int.from_bytes(digest[:4], "big")
        score = 0.5 + (n / 0xFFFFFFFF) * 0.5
        out.append({
            "id": gate["id"],
            "score": round(score, 4),
            "dataset_revision": "synthetic@2026-07",
            "source_revision": "synthetic-rev-0",
            "captured_at_unix_ms": int(time.time() * 1000),
            "note": "deterministic synthetic evidence (no benchmark harness wired)",
        })
    return out


def _load_cached_record(kernel_id: str) -> dict[str, Any] | None:
    """Return the cached record if present, else ``None``.

    A cached record is reused as the candidate's evidence pool so a
    subsequent ``promote`` with stricter gates sees the same scores.
    """
    path = promotion_path(kernel_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def build_candidate(
    kernel_id: str,
    gates: list[dict[str, Any]],
    cached: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build an in-memory candidate PromotionRecord from gates + cached evidence.

    When a cached record exists we reuse its ``evidence`` field verbatim
    (so the gate scores are stable across runs). Otherwise we synthesize
    deterministic evidence.

    The returned dict has the canonical field order; ``signature`` and
    ``content_hash`` are populated by the caller.
    """
    if cached is not None and isinstance(cached.get("evidence"), list):
        evidence = list(cached["evidence"])
    else:
        evidence = _synthetic_evidence(kernel_id, gates)

    # Carry the source_revision forward if the cache had one.
    src_rev = "synthetic-rev-0"
    if cached and isinstance(cached.get("source_revision"), str):
        src_rev = cached["source_revision"]

    tuning_id = None
    if cached and isinstance(cached.get("tuning_record_id"), str):
        tuning_id = cached["tuning_record_id"]

    justification = ""
    if cached and isinstance(cached.get("justification"), str):
        justification = cached["justification"]

    return {
        "candidate_id": kernel_id,
        "source_revision": src_rev,
        "approved_at_unix_ms": int(time.time() * 1000),
        "approver": "",
        "gates": gates,
        "evidence": evidence,
        "justification": justification,
        "tuning_record_id": tuning_id,
        "signature": None,
        "content_hash": "",
    }


# ---------------------------------------------------------------------------
# Validation (mirrors PromotionRecord::validate + PromotionValidator::promote).
# ---------------------------------------------------------------------------

class PromotionError(Exception):
    """Raised when a record fails promotion. Holds a structured payload."""

    def __init__(self, kind: str, **fields: Any) -> None:
        super().__init__(kind)
        self.kind = kind
        self.fields = fields

    def to_dict(self) -> dict[str, Any]:
        out = {"error": "promotion_rejected", "kind": self.kind}
        out.update(self.fields)
        return out


def validate(record: dict[str, Any]) -> None:
    """Raise ``PromotionError`` on the first failing gate; return ``None`` on success.

    Mirrors ``PromotionRecord::validate`` in quality.rs: gates must be
    non-empty, every gate must have matching evidence, and the evidence
    score must pass under the gate's direction.
    """
    gates = record.get("gates") or []
    if not gates:
        raise PromotionError("promotion_without_gates")

    evidence = record.get("evidence") or []
    by_id: dict[str, dict[str, Any]] = {}
    for ev in evidence:
        if not isinstance(ev, dict) or not isinstance(ev.get("id"), str):
            continue
        if ev["id"] in by_id:
            raise PromotionError("duplicate_evidence", gate=ev["id"])
        by_id[ev["id"]] = ev

    for gate in gates:
        gid = gate.get("id")
        ev = by_id.get(gid) if isinstance(gid, str) else None
        if ev is None:
            raise PromotionError("gate_missing_evidence", gate=gid)
        score = ev.get("score")
        if not isinstance(score, (int, float)):
            raise PromotionError("gate_missing_evidence", gate=gid)
        if not gate_passes(gate, float(score)):
            raise PromotionError(
                "gate_rejected",
                gate=gid,
                observed=float(score),
                threshold=float(gate["threshold"]),
                direction=gate["direction"],
            )
    return None


# ---------------------------------------------------------------------------
# I/O.
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)


def write_promotion_record(kernel_id: str, record: dict[str, Any]) -> str:
    """Persist the finalized record to disk; return the absolute path."""
    path = promotion_path(kernel_id)
    _ensure_dir(path)
    payload = json.dumps(record, indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload + "\n")
    return path


# ---------------------------------------------------------------------------
# Summary renderer.
# ---------------------------------------------------------------------------

def _human_summary(record: dict[str, Any], cache_path: str, decision: str) -> str:
    lines = [
        f"candidate_id      : {record['candidate_id']}",
        f"source_revision   : {record['source_revision']}",
        f"approver          : {record['approver'] or '<unset>'}",
        f"decision          : {decision}",
        f"approved_at_unix_ms: {record['approved_at_unix_ms']}",
        f"gates             : {len(record['gates'])}",
        f"evidence          : {len(record['evidence'])}",
        f"content_hash      : {record['content_hash'][:16]}...",
        f"signature         : {(record.get('signature') or '<unsigned>')[:16]}"
        f"{'...' if record.get('signature') else ''}",
        f"cache_path        : {cache_path}",
        "promotion         : OK",
    ]
    return "\n".join(lines)


def _json_summary(
    record: dict[str, Any],
    cache_path: str,
    decision: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "decision": decision,
        "candidate_id": record["candidate_id"],
        "source_revision": record["source_revision"],
        "approver": record["approver"],
        "approved_at_unix_ms": record["approved_at_unix_ms"],
        "gate_count": len(record["gates"]),
        "evidence_count": len(record["evidence"]),
        "gates": record["gates"],
        "evidence": record["evidence"],
        "content_hash": record["content_hash"],
        "signature": record.get("signature"),
        "cache_path": cache_path,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def cmd_promote(args: argparse.Namespace) -> int:
    """CLI entry point: ``promote <kernel_id> --gates ... [--sign-key ...] [--json]``"""
    try:
        gates = parse_gates(args.gates)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    cached = _load_cached_record(args.kernel_id)
    candidate = build_candidate(args.kernel_id, gates, cached)

    # Validate before signing/hashing so a bad gate doesn't bloat the cache.
    try:
        validate(candidate)
    except PromotionError as e:
        payload = e.to_dict()
        if getattr(args, "json", False):
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            gate = payload.get("gate", "?")
            if e.kind == "gate_rejected":
                msg = (
                    f"promotion rejected: gate {gate!r} "
                    f"observed={payload['observed']:.4f} "
                    f"threshold={payload['threshold']:.4f}"
                )
            elif e.kind == "gate_missing_evidence":
                msg = f"promotion rejected: gate {gate!r} has no evidence"
            elif e.kind == "duplicate_evidence":
                msg = f"promotion rejected: duplicate evidence for gate {gate!r}"
            else:
                msg = f"promotion rejected: {e.kind}"
            print(f"error: {msg}", file=sys.stderr)
        return 2

    # Sign + hash. Approver is taken from --approver or $USER / 'unknown'.
    approver = getattr(args, "approver", None) or os.environ.get("USER") or "unknown"
    candidate["approver"] = approver

    if args.sign_key:
        try:
            key_bytes = bytes.fromhex(args.sign_key)
        except ValueError:
            print(
                f"error: --sign-key must be hex bytes, got {args.sign_key!r}",
                file=sys.stderr,
            )
            return 2
        if not key_bytes:
            print("error: --sign-key must not be empty", file=sys.stderr)
            return 2
        candidate["signature"] = sign_record(candidate, key_bytes)
    else:
        candidate["signature"] = None

    candidate["content_hash"] = content_hash(candidate)

    try:
        path = write_promotion_record(args.kernel_id, candidate)
    except OSError as e:
        print(f"error: could not write promotion cache: {e}", file=sys.stderr)
        return 3

    decision = getattr(args, "decision", None) or "auto"
    if getattr(args, "json", False):
        sys.stdout.write(
            json.dumps(_json_summary(candidate, path, decision), indent=2, sort_keys=True) + "\n"
        )
    else:
        print(_human_summary(candidate, path, decision))
    return 0

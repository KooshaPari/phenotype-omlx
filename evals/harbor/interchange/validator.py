"""Eval Interchange Contract v1.0 — validation rules."""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

from .contract import EvalReport

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating an EvalReport against contract rules."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _canonical_json(doc: dict[str, Any]) -> bytes:
    """Canonical JSON: sorted keys, no whitespace, UTF-8."""
    return json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _compute_top_level_sha256(doc: dict[str, Any]) -> str:
    """Compute SHA-256 over canonical JSON of the document minus hash_chain."""
    payload = {k: v for k, v in doc.items() if k != "hash_chain"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _compute_task_ids_sorted_sha256(report: EvalReport) -> str:
    """Collect task_ids from all suites, sort, join with newline, SHA-256."""
    task_ids: list[str] = []
    for suite in report.suites:
        if hasattr(suite, "task_ids"):
            raw = suite.task_ids
            if isinstance(raw, list):
                task_ids.extend(str(t) for t in raw)
    task_ids.sort()
    joined = "\n".join(task_ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def validate(
    report: EvalReport, raw_doc: dict[str, Any] | None = None
) -> ValidationResult:
    """Validate an EvalReport against all contract rules.

    Rules:
        R-VERSION  — contract_version must be "1.0"
        R-HASHCHAIN — hash_chain verification
        R-PRODUCER — producer block must have required fields
        R-SUITES   — suites must be non-empty
        R-TOTALS   — totals must be present
        W-EVIDENCE — warn if any evidence_label != live_verified
    """
    result = ValidationResult(valid=True)
    errors = result.errors
    warnings = result.warnings

    # ── R-VERSION ──────────────────────────────────────────────
    if report.contract_version != "1.0":
        errors.append(
            f"R-VERSION: contract_version must be '1.0', got '{report.contract_version}'"
        )

    # ── R-PRODUCER ────────────────────────────────────────────
    producer = report.producer
    if not producer.name:
        errors.append("R-PRODUCER: producer.name is required")
    if not producer.version:
        errors.append("R-PRODUCER: producer.version is required")
    if not producer.commit_sha:
        errors.append("R-PRODUCER: producer.commit_sha is required")

    # ── R-SUITES ──────────────────────────────────────────────
    if not report.suites:
        errors.append("R-SUITES: suites must not be empty")

    # ── R-TOTALS ──────────────────────────────────────────────
    if report.totals is None:
        errors.append("R-TOTALS: totals block is required")

    # ── W-EVIDENCE ────────────────────────────────────────────
    for suite in report.suites:
        if suite.evidence_label.value != "live_verified":
            warnings.append(
                f"W-EVIDENCE: suite '{suite.suite}' evidence_label is "
                f"'{suite.evidence_label.value}', expected 'live_verified'"
            )

    # ── R-HASHCHAIN ───────────────────────────────────────────
    if raw_doc is not None:
        expected_top = _compute_top_level_sha256(raw_doc)
        actual_top = report.hash_chain.top_level_sha256
        if expected_top != actual_top:
            errors.append(
                f"R-HASHCHAIN: top_level_sha256 mismatch — "
                f"expected {expected_top[:16]}…, got {actual_top[:16]}…"
            )

        expected_task = _compute_task_ids_sorted_sha256(report)
        actual_task = report.hash_chain.task_ids_sorted_sha256
        if expected_task != actual_task:
            errors.append(
                f"R-HASHCHAIN: task_ids_sorted_sha256 mismatch — "
                f"expected {expected_task[:16]}…, got {actual_task[:16]}…"
            )
    else:
        warnings.append(
            "R-HASHCHAIN: raw_doc not provided, hash chain verification skipped"
        )

    # ── finalize ──────────────────────────────────────────────
    if warnings:
        for w in warnings:
            logger.warning(w)

    if errors:
        result.valid = False

    return result

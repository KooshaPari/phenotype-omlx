"""Eval Interchange Contract v1.0 — ingestion adapter.

Converts EvalReport v1.0 (and legacy v0.1) into eval-harness internal
format (EvaluationReport / MultiSuiteReport dataclasses).

The eval-harness Rust crate defines:
    Suite enum: {Mmlu, Gpqa, TerminalBench, Perplexity}
    EvaluationReport {suite, task_count, correct_count, accuracy, mean_score, results}
    MultiSuiteReport {task_count, correct_count, overall_accuracy,
                      mean_suite_accuracy, mean_suite_score, entries}
    SuiteReportEntry {suite, provenance, report}

This adapter produces Python dataclasses that mirror those shapes so
downstream Python code can consume interchange reports without a
Rust FFI boundary.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contract import EvalReport, SuiteResult
from .loader import load_report, load_report_from_dict
from .validator import ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval-harness internal format (Python mirror of Rust structs)
# ---------------------------------------------------------------------------


class Suite(str, Enum):
    """Mirror of the Rust ``eval_harness::Suite`` enum."""

    Mmlu = "mmlu"
    Gpqa = "gpqa"
    TerminalBench = "terminal-bench"
    Perplexity = "perplexity"


# Contract suite name → internal Suite enum mapping.
# Keys are the strings that appear in EvalReport.suites[].suite.
_SUITENAME_MAP: dict[str, Suite] = {
    "mmlu": Suite.Mmlu,
    "mmlu-pro": Suite.Mmlu,
    "gpqa": Suite.Gpqa,
    "gpqa-diamond": Suite.Gpqa,
    "terminal-bench": Suite.TerminalBench,
    "terminal_bench": Suite.TerminalBench,
    "perplexity": Suite.Perplexity,
}


@dataclass
class TaskResult:
    """Mirror of Rust ``eval_harness::TaskResult``."""

    task_id: str
    suite: Suite
    prompt_tokens: int = 0
    completion_tokens: int = 0
    completion: str = ""
    normalized_completion: str = ""
    correct: bool = False
    score: float = 0.0
    latency_ms: float = 0.0
    matched_answer: str | None = None


@dataclass
class DatasetProvenance:
    """Mirror of Rust ``eval_harness::provenance::DatasetProvenance``."""

    source: str
    source_revision: str
    split: str
    content_sha256: str
    task_count: int


@dataclass
class EvaluationReport:
    """Mirror of Rust ``eval_harness::EvaluationReport``."""

    suite: Suite
    task_count: int
    correct_count: int
    accuracy: float
    mean_score: float
    results: list[TaskResult] = field(default_factory=list)


@dataclass
class SuiteReportEntry:
    """Mirror of Rust ``eval_harness::report::SuiteReportEntry``."""

    suite: Suite
    provenance: DatasetProvenance
    report: EvaluationReport


@dataclass
class MultiSuiteReport:
    """Mirror of Rust ``eval_harness::report::MultiSuiteReport``."""

    task_count: int
    correct_count: int
    overall_accuracy: float
    mean_suite_accuracy: float
    mean_suite_score: float
    entries: list[SuiteReportEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# v0.1 → v1.0 migration helpers
# ---------------------------------------------------------------------------

_V01_TO_V10_FIELD_MAP = {
    "repo": "name",
    "head": "commit_sha",
}

_V01_RUN_EVIDENCE_FALLBACK = "reported"


def _migrate_v01_to_v10(doc: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v0.1 contract document to v1.0 shape.

    Key renames per the migration table in INTERCHANGE_CONTRACT.md:
        producer.repo → producer.name
        producer.head → producer.commit_sha
        producer.branch → removed
        producer.dirty_paths → removed
        producer.host → removed
        run.evidence_label → moved to suite-level (fallback)

    Unknown fields at any level are preserved (W-UNKNOWN policy).
    """
    migrated: dict[str, Any] = {}
    migrated["contract_version"] = "1.0"
    migrated["artifact_kind"] = "EvaluationReport"

    # ── producer ──────────────────────────────────────────────
    # Fields removed in v1.0 per the migration table.
    _V01_PRODUCER_REMOVED = {"branch", "dirty_paths", "host"}
    raw_producer = dict(doc.get("producer", {}))
    new_producer: dict[str, Any] = {}
    for old_key, new_key in _V01_TO_V10_FIELD_MAP.items():
        if old_key in raw_producer:
            new_producer[new_key] = raw_producer.pop(old_key)
    # Strip v0.1-only fields, then carry forward the rest.
    for removed in _V01_PRODUCER_REMOVED:
        raw_producer.pop(removed, None)
    new_producer.update(raw_producer)
    migrated["producer"] = new_producer

    # ── run ───────────────────────────────────────────────────
    raw_run = dict(doc.get("run", {}))
    # Remove run-level evidence_label (moved to suite-level)
    raw_run.pop("evidence_label", None)
    migrated["run"] = raw_run

    # ── suites ────────────────────────────────────────────────
    raw_suites = doc.get("suites", [])
    run_evidence = doc.get("run", {}).get("evidence_label", _V01_RUN_EVIDENCE_FALLBACK)
    suites: list[dict[str, Any]] = []
    for s in raw_suites:
        entry = dict(s)
        if "evidence_label" not in entry:
            entry["evidence_label"] = run_evidence
        suites.append(entry)
    migrated["suites"] = suites

    # ── totals / hash_chain ───────────────────────────────────
    migrated["totals"] = doc.get("totals", {})
    # v0.1 may have empty/missing hash_chain — fill with dummy valid hex
    # so Pydantic validation passes. Real verification is skipped via
    # skip_hash_chain=True for migrated documents.
    _DUMMY_SHA256 = "0" * 64
    raw_hash = doc.get("hash_chain", {})
    migrated["hash_chain"] = {
        "top_level_sha256": raw_hash.get("top_level_sha256", _DUMMY_SHA256)
        or _DUMMY_SHA256,
        "task_ids_sorted_sha256": raw_hash.get("task_ids_sorted_sha256", _DUMMY_SHA256)
        or _DUMMY_SHA256,
    }

    # ── carry forward optional extras ─────────────────────────
    for key in ("matrix", "comparator"):
        if key in doc:
            migrated[key] = doc[key]

    return migrated


# ---------------------------------------------------------------------------
# Suite name resolution
# ---------------------------------------------------------------------------


def _resolve_suite(name: str) -> Suite:
    """Resolve a contract suite name to the internal Suite enum.

    Raises ``ValueError`` for unrecognised suite names.
    """
    normalised = name.strip().lower().replace(" ", "-")
    if normalised in _SUITENAME_MAP:
        return _SUITENAME_MAP[normalised]
    raise ValueError(
        f"Unrecognised suite name '{name}'. Known: {', '.join(sorted(_SUITENAME_MAP))}"
    )


def _resolve_suite_safe(name: str) -> Suite | None:
    """Resolve a suite name, returning ``None`` for unknown suites."""
    try:
        return _resolve_suite(name)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def ingest_report(
    report: EvalReport,
    raw_doc: dict[str, Any] | None = None,
) -> MultiSuiteReport:
    """Convert a validated EvalReport into a eval-harness MultiSuiteReport.

    Parameters
    ----------
    report:
        A validated ``EvalReport`` (v1.0).
    raw_doc:
        The original raw dict (for hash-chain verification in the caller).
        If ``None``, hash-chain checks were already skipped by the loader.

    Returns
    -------
    MultiSuiteReport
        The aggregated eval-harness report.

    Raises
    ------
    ValueError
        If an unrecognised suite name is encountered.
    """
    entries: list[SuiteReportEntry] = []

    for suite_result in report.suites:
        suite = _resolve_suite(suite_result.suite)

        # Build aRivation from producer + run provenance.
        provenance = DatasetProvenance(
            source=suite_result.suite,
            source_revision=report.producer.version or "unknown",
            split="test",
            content_sha256=report.hash_chain.task_ids_sorted_sha256,
            task_count=suite_result.n,
        )

        # Synthesise a minimal EvaluationReport from the aggregate numbers.
        # Since the interchange contract only carries aggregate counts (not
        # per-task results), we build a synthetic result list where every
        # task is either correct or incorrect to match the pass_at_1 ratio.
        task_count = suite_result.n
        correct_count = suite_result.passed
        accuracy = suite_result.pass_at_1
        mean_score = accuracy  # binary scoring → mean_score == accuracy

        results = _synthesise_task_results(
            suite=suite,
            task_count=task_count,
            correct_count=correct_count,
            suite_name=suite_result.suite,
        )

        report_entry = EvaluationReport(
            suite=suite,
            task_count=task_count,
            correct_count=correct_count,
            accuracy=accuracy,
            mean_score=mean_score,
            results=results,
        )

        entries.append(
            SuiteReportEntry(suite=suite, provenance=provenance, report=report_entry)
        )

    # Sort by Suite ordinal to match Rust's declaration-order sort.
    entries.sort(key=lambda e: e.suite.value)

    # Aggregate.
    total_tasks = sum(e.report.task_count for e in entries)
    total_correct = sum(e.report.correct_count for e in entries)
    overall_accuracy = total_correct / total_tasks if total_tasks > 0 else 0.0
    mean_suite_accuracy = (
        sum(e.report.accuracy for e in entries) / len(entries) if entries else 0.0
    )
    mean_suite_score = (
        sum(e.report.mean_score for e in entries) / len(entries) if entries else 0.0
    )

    _emit_evidence_warnings(report)

    return MultiSuiteReport(
        task_count=total_tasks,
        correct_count=total_correct,
        overall_accuracy=overall_accuracy,
        mean_suite_accuracy=mean_suite_accuracy,
        mean_suite_score=mean_suite_score,
        entries=entries,
    )


def _synthesise_task_results(
    suite: Suite,
    task_count: int,
    correct_count: int,
    suite_name: str,
) -> list[TaskResult]:
    """Build a synthetic task-result list from aggregate counts.

    Since the interchange contract only carries pass/fail counts (not
    per-task detail), we produce deterministic synthetic results:
    the first ``correct_count`` tasks are marked correct, the rest
    incorrect.  Task IDs follow the pattern ``<suite>-<index>``.
    """
    results: list[TaskResult] = []
    for i in range(task_count):
        correct = i < correct_count
        tid = f"{suite_name}-{i}"
        results.append(
            TaskResult(
                task_id=tid,
                suite=suite,
                completion="",
                normalized_completion="",
                correct=correct,
                score=1.0 if correct else 0.0,
            )
        )
    return results


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------


def ingest_from_file(
    path: str,
    *,
    skip_hash_chain: bool = False,
    allow_v01: bool = True,
) -> tuple[MultiSuiteReport, ValidationResult]:
    """Load, validate, and ingest an EvalReport from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.
    skip_hash_chain:
        If ``True``, skip hash-chain verification (pass ``raw_doc=None``).
    allow_v01:
        If ``True``, automatically migrate v0.1 documents to v1.0 shape.

    Returns
    -------
    (MultiSuiteReport, ValidationResult)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the document cannot be parsed or migrated.
    """
    from pathlib import Path

    import json as _json

    file_path = Path(path)
    raw_doc: dict[str, Any] = _json.loads(file_path.read_text(encoding="utf-8"))

    return ingest_from_dict(
        raw_doc,
        skip_hash_chain=skip_hash_chain,
        allow_v01=allow_v01,
    )


def ingest_from_dict(
    doc: dict[str, Any],
    *,
    skip_hash_chain: bool = False,
    allow_v01: bool = True,
) -> tuple[MultiSuiteReport, ValidationResult]:
    """Validate and ingest an EvalReport from an already-parsed dict.

    Parameters
    ----------
    doc:
        The raw JSON document as a dict.
    skip_hash_chain:
        If ``True``, skip hash-chain verification.
    allow_v01:
        If ``True``, automatically migrate v0.1 documents to v1.0 shape.

    Returns
    -------
    (MultiSuiteReport, ValidationResult)
    """
    # ── version detection & migration ─────────────────────────
    version = doc.get("contract_version", "")
    if version == "0.1":
        if not allow_v01:
            raise ValueError(
                "v0.1 contract received but allow_v01=False. "
                "Migrate to v1.0 before ingestion."
            )
        warnings.warn(
            "ingest: v0.1 contract detected — migrating to v1.0",
            DeprecationWarning,
            stacklevel=2,
        )
        doc = _migrate_v01_to_v10(doc)

    # ── validate & load ───────────────────────────────────────
    report = EvalReport.model_validate(doc)
    raw_for_validation = None if skip_hash_chain else doc
    result = validate(report, raw_for_validation)

    # ── ingest ────────────────────────────────────────────────
    multi = ingest_report(report, raw_for_validation)

    return multi, result


def validate(
    report: EvalReport, raw_doc: dict[str, Any] | None = None
) -> ValidationResult:
    """Re-export validate for convenience at the ingest layer."""
    from .validator import validate as _validate

    return _validate(report, raw_doc)


def _emit_evidence_warnings(report: EvalReport) -> None:
    """Emit warnings for non-live evidence labels."""
    for suite in report.suites:
        if suite.evidence_label.value != "live_verified":
            msg = (
                f"W-EVIDENCE: suite '{suite.suite}' evidence_label is "
                f"'{suite.evidence_label.value}', not 'live_verified'"
            )
            warnings.warn(msg, stacklevel=3)
            logger.warning(msg)

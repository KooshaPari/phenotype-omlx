"""Unit tests for Eval Interchange Contract v1.0 ingestion adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from pathlib import Path

import pytest

from .contract import EvalReport, HashChain, ProducerInfo, RunInfo, SuiteResult, Totals
from .ingest import (
    DatasetProvenance,
    EvaluationReport,
    MultiSuiteReport,
    Suite,
    SuiteReportEntry,
    TaskResult,
    _migrate_v01_to_v10,
    _resolve_suite,
    _resolve_suite_safe,
    _synthesise_task_results,
    ingest_from_dict,
    ingest_from_file,
    ingest_report,
)
from .validator import _canonical_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_doc(
    *,
    contract_version: str = "1.0",
    suites: list[dict] | None = None,
    producer: dict | None = None,
    totals: dict | None = None,
    hash_chain: dict | None = None,
) -> dict:
    """Build a raw dict matching the v1.0 contract schema for testing."""
    if suites is None:
        suites = [
            {
                "suite": "mmlu-pro",
                "n": 25,
                "passed": 25,
                "pass_at_1": 1.0,
                "evidence_label": "live_verified",
            }
        ]
    if producer is None:
        producer = {
            "name": "pheno-harness",
            "version": "5.0.0",
            "commit_sha": "a" * 40,
        }
    if totals is None:
        totals = {"cells": 25, "passed": 25, "pass_at_1": 1.0}
    if hash_chain is None:
        tmp = {
            "contract_version": contract_version,
            "artifact_kind": "EvaluationReport",
            "producer": producer,
            "run": {
                "run_id": "00000000-0000-0000-0000-000000000001",
                "started_at": "2026-01-01T00:00:00Z",
                "model": "Qwen/Qwen3.5-0.8B",
                "variant": "stock",
                "judge_mode": "deterministic",
            },
            "suites": suites,
            "totals": totals,
        }
        top_hash = hashlib.sha256(_canonical_json(tmp)).hexdigest()
        task_ids = sorted(
            str(tid) for s in suites for tid in [s.get("task_id")] if tid is not None
        )
        task_hash = hashlib.sha256("\n".join(task_ids).encode()).hexdigest()
        hash_chain = {
            "top_level_sha256": top_hash,
            "task_ids_sorted_sha256": task_hash,
        }
    return {
        "contract_version": contract_version,
        "artifact_kind": "EvaluationReport",
        "producer": producer,
        "run": {
            "run_id": "00000000-0000-0000-0000-000000000001",
            "started_at": "2026-01-01T00:00:00Z",
            "model": "Qwen/Qwen3.5-0.8B",
            "variant": "stock",
            "judge_mode": "deterministic",
        },
        "suites": suites,
        "totals": totals,
        "hash_chain": hash_chain,
    }


def _make_v01_doc() -> dict:
    """Build a raw v0.1 contract document."""
    return {
        "contract_version": "0.1",
        "artifact_kind": "EvaluationReport",
        "producer": {
            "repo": "pheno-harness",
            "version": "4.0.0",
            "head": "b" * 40,
            "branch": "main",
            "dirty_paths": ["src/foo.py"],
            "host": "build-01",
        },
        "run": {
            "run_id": "00000000-0000-0000-0000-000000000002",
            "started_at": "2025-12-01T00:00:00Z",
            "model": "Qwen/Qwen3-0.5B",
            "variant": "stock",
            "judge_mode": "deterministic",
            "evidence_label": "reported",
        },
        "suites": [
            {
                "suite": "mmlu",
                "n": 10,
                "passed": 7,
                "pass_at_1": 0.7,
            }
        ],
        "totals": {"cells": 10, "passed": 7, "pass_at_1": 0.7},
        "hash_chain": {
            "top_level_sha256": "",
            "task_ids_sorted_sha256": "",
        },
    }


# ---------------------------------------------------------------------------
# Suite resolution
# ---------------------------------------------------------------------------


class TestSuiteResolution:
    def test_resolve_mmlu_pro(self):
        assert _resolve_suite("mmlu-pro") == Suite.Mmlu

    def test_resolve_mmlu_lower(self):
        assert _resolve_suite("mmlu") == Suite.Mmlu

    def test_resolve_gpqa(self):
        assert _resolve_suite("gpqa") == Suite.Gpqa

    def test_resolve_gpqa_diamond(self):
        assert _resolve_suite("gpqa-diamond") == Suite.Gpqa

    def test_resolve_terminal_bench(self):
        assert _resolve_suite("terminal-bench") == Suite.TerminalBench

    def test_resolve_perplexity(self):
        assert _resolve_suite("perplexity") == Suite.Perplexity

    def test_resolve_case_insensitive(self):
        assert _resolve_suite("MMLU-PRO") == Suite.Mmlu

    def test_resolve_strips_whitespace(self):
        assert _resolve_suite("  mmlu-pro  ") == Suite.Mmlu

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unrecognised suite name"):
            _resolve_suite("bogus-suite")

    def test_resolve_safe_returns_none_for_unknown(self):
        assert _resolve_suite_safe("unknown-suite") is None

    def test_resolve_safe_returns_suite_for_known(self):
        assert _resolve_suite_safe("mmlu") == Suite.Mmlu


# ---------------------------------------------------------------------------
# Task result synthesis
# ---------------------------------------------------------------------------


class TestSynthesiseTaskResults:
    def test_all_correct(self):
        results = _synthesise_task_results(Suite.Mmlu, 3, 3, "mmlu")
        assert len(results) == 3
        assert all(r.correct for r in results)
        assert all(r.score == 1.0 for r in results)

    def test_none_correct(self):
        results = _synthesise_task_results(Suite.Gpqa, 3, 0, "gpqa")
        assert len(results) == 3
        assert all(not r.correct for r in results)
        assert all(r.score == 0.0 for r in results)

    def test_partial_correct(self):
        results = _synthesise_task_results(Suite.Mmlu, 5, 3, "mmlu-pro")
        assert len(results) == 5
        correct = [r for r in results if r.correct]
        incorrect = [r for r in results if not r.correct]
        assert len(correct) == 3
        assert len(incorrect) == 2

    def test_task_ids_follow_pattern(self):
        results = _synthesise_task_results(Suite.Mmlu, 3, 2, "mmlu-pro")
        ids = [r.task_id for r in results]
        assert ids == ["mmlu-pro-0", "mmlu-pro-1", "mmlu-pro-2"]

    def test_suite_tagged_on_each_result(self):
        results = _synthesise_task_results(Suite.Gpqa, 2, 1, "gpqa")
        for r in results:
            assert r.suite == Suite.Gpqa

    def test_zero_tasks(self):
        results = _synthesise_task_results(Suite.Mmlu, 0, 0, "mmlu")
        assert results == []


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


class TestIngestReport:
    def test_single_suite(self):
        raw = _make_raw_doc()
        report = EvalReport.model_validate(raw)
        multi = ingest_report(report)

        assert isinstance(multi, MultiSuiteReport)
        assert multi.task_count == 25
        assert multi.correct_count == 25
        assert multi.overall_accuracy == 1.0
        assert len(multi.entries) == 1
        assert multi.entries[0].suite == Suite.Mmlu
        assert multi.entries[0].report.task_count == 25
        assert multi.entries[0].report.correct_count == 25
        assert multi.entries[0].report.accuracy == 1.0

    def test_multi_suite_aggregation(self):
        raw = _make_raw_doc(
            suites=[
                {
                    "suite": "mmlu-pro",
                    "n": 20,
                    "passed": 18,
                    "pass_at_1": 0.9,
                    "evidence_label": "live_verified",
                },
                {
                    "suite": "gpqa-diamond",
                    "n": 10,
                    "passed": 7,
                    "pass_at_1": 0.7,
                    "evidence_label": "live_verified",
                },
            ],
            totals={"cells": 30, "passed": 25, "pass_at_1": 0.833},
        )
        report = EvalReport.model_validate(raw)
        multi = ingest_report(report)

        assert multi.task_count == 30
        assert multi.correct_count == 25
        assert multi.overall_accuracy == pytest.approx(25 / 30)
        assert multi.mean_suite_accuracy == pytest.approx((0.9 + 0.7) / 2)
        assert len(multi.entries) == 2
        # Entries sorted by suite value.
        assert multi.entries[0].suite == Suite.Gpqa
        assert multi.entries[1].suite == Suite.Mmlu

    def test_entries_sorted_deterministically(self):
        raw = _make_raw_doc(
            suites=[
                {
                    "suite": "gpqa",
                    "n": 5,
                    "passed": 5,
                    "pass_at_1": 1.0,
                    "evidence_label": "live_verified",
                },
                {
                    "suite": "mmlu-pro",
                    "n": 10,
                    "passed": 9,
                    "pass_at_1": 0.9,
                    "evidence_label": "live_verified",
                },
            ],
            totals={"cells": 15, "passed": 14, "pass_at_1": 0.933},
        )
        report = EvalReport.model_validate(raw)
        multi = ingest_report(report)

        assert multi.entries[0].suite == Suite.Gpqa
        assert multi.entries[1].suite == Suite.Mmlu

    def test_provenance_populated(self):
        raw = _make_raw_doc()
        report = EvalReport.model_validate(raw)
        multi = ingest_report(report)

        prov = multi.entries[0].provenance
        assert isinstance(prov, DatasetProvenance)
        assert prov.source == "mmlu-pro"
        assert prov.source_revision == "5.0.0"
        assert prov.split == "test"
        assert prov.task_count == 25

    def test_synthesised_results_present(self):
        raw = _make_raw_doc()
        report = EvalReport.model_validate(raw)
        multi = ingest_report(report)

        entry = multi.entries[0]
        assert len(entry.report.results) == 25
        assert all(isinstance(r, TaskResult) for r in entry.report.results)
        assert all(r.correct for r in entry.report.results)

    def test_unknown_suite_raises(self):
        raw = _make_raw_doc(
            suites=[
                {
                    "suite": "bogus-suite",
                    "n": 5,
                    "passed": 3,
                    "pass_at_1": 0.6,
                    "evidence_label": "live_verified",
                }
            ]
        )
        report = EvalReport.model_validate(raw)
        with pytest.raises(ValueError, match="Unrecognised suite name"):
            ingest_report(report)


# ---------------------------------------------------------------------------
# v0.1 migration
# ---------------------------------------------------------------------------


class TestV01Migration:
    def test_producer_repo_renamed(self):
        v01 = _make_v01_doc()
        migrated = _migrate_v01_to_v10(v01)
        assert migrated["contract_version"] == "1.0"
        assert migrated["producer"]["name"] == "pheno-harness"
        assert "repo" not in migrated["producer"]

    def test_producer_head_renamed(self):
        v01 = _make_v01_doc()
        migrated = _migrate_v01_to_v10(v01)
        assert migrated["producer"]["commit_sha"] == "b" * 40
        assert "head" not in migrated["producer"]

    def test_branch_removed(self):
        v01 = _make_v01_doc()
        migrated = _migrate_v01_to_v10(v01)
        assert "branch" not in migrated["producer"]

    def test_dirty_paths_removed(self):
        v01 = _make_v01_doc()
        migrated = _migrate_v01_to_v10(v01)
        assert "dirty_paths" not in migrated["producer"]

    def test_host_removed(self):
        v01 = _make_v01_doc()
        migrated = _migrate_v01_to_v10(v01)
        assert "host" not in migrated["producer"]

    def test_run_evidence_label_removed(self):
        v01 = _make_v01_doc()
        migrated = _migrate_v01_to_v10(v01)
        assert "evidence_label" not in migrated["run"]

    def test_suites_get_evidence_from_run(self):
        v01 = _make_v01_doc()
        migrated = _migrate_v01_to_v10(v01)
        assert migrated["suites"][0]["evidence_label"] == "reported"

    def test_suites_keep_existing_evidence(self):
        v01 = _make_v01_doc()
        v01["suites"][0]["evidence_label"] = "live_verified"
        migrated = _migrate_v01_to_v10(v01)
        assert migrated["suites"][0]["evidence_label"] == "live_verified"

    def test_matrix_comparator_preserved(self):
        v01 = _make_v01_doc()
        v01["matrix"] = {"x": 1}
        v01["comparator"] = {"y": 2}
        migrated = _migrate_v01_to_v10(v01)
        assert migrated["matrix"] == {"x": 1}
        assert migrated["comparator"] == {"y": 2}

    def test_full_v01_ingestion(self):
        v01 = _make_v01_doc()
        multi, result = ingest_from_dict(v01, allow_v01=True, skip_hash_chain=True)
        assert multi.task_count == 10
        assert multi.correct_count == 7
        assert multi.overall_accuracy == pytest.approx(0.7)
        # v0.1 migration produces a v1.0 document that passes validation.
        assert result.valid

    def test_v01_rejected_when_not_allowed(self):
        v01 = _make_v01_doc()
        with pytest.raises(ValueError, match="v0.1 contract received"):
            ingest_from_dict(v01, allow_v01=False)


# ---------------------------------------------------------------------------
# Evidence warnings
# ---------------------------------------------------------------------------


class TestEvidenceWarnings:
    def test_live_verified_no_warning(self):
        raw = _make_raw_doc()
        report = EvalReport.model_validate(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            multi = ingest_report(report)
            assert multi is not None

    def test_reported_emits_warning(self):
        raw = _make_raw_doc()
        raw["suites"][0]["evidence_label"] = "reported"
        payload = {k: v for k, v in raw.items() if k != "hash_chain"}
        raw["hash_chain"]["top_level_sha256"] = hashlib.sha256(
            _canonical_json(payload)
        ).hexdigest()
        raw["hash_chain"]["task_ids_sorted_sha256"] = hashlib.sha256(b"").hexdigest()
        report = EvalReport.model_validate(raw)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ingest_report(report)
            evidence_warnings = [x for x in w if "W-EVIDENCE" in str(x.message)]
            assert len(evidence_warnings) == 1
            assert "reported" in str(evidence_warnings[0].message)

    def test_synthetic_emits_warning(self):
        raw = _make_raw_doc()
        raw["suites"][0]["evidence_label"] = "synthetic"
        payload = {k: v for k, v in raw.items() if k != "hash_chain"}
        raw["hash_chain"]["top_level_sha256"] = hashlib.sha256(
            _canonical_json(payload)
        ).hexdigest()
        raw["hash_chain"]["task_ids_sorted_sha256"] = hashlib.sha256(b"").hexdigest()
        report = EvalReport.model_validate(raw)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ingest_report(report)
            evidence_warnings = [x for x in w if "synthetic" in str(x.message)]
            assert len(evidence_warnings) == 1


# ---------------------------------------------------------------------------
# ingest_from_dict
# ---------------------------------------------------------------------------


class TestIngestFromDict:
    def test_v10_passthrough(self):
        raw = _make_raw_doc()
        multi, result = ingest_from_dict(raw, skip_hash_chain=True)
        assert result.valid
        assert multi.task_count == 25

    def test_v10_with_validation_errors(self):
        raw = _make_raw_doc()
        report = EvalReport.model_validate(raw)
        report.contract_version = "1.1"
        # ingest_from_dict validates internally; we test via direct call.
        # For this test we just verify ingest_report works with the object.
        raw["contract_version"] = "1.1"
        with pytest.raises(Exception):
            EvalReport.model_validate(raw)

    def test_skip_hash_chain(self):
        raw = _make_raw_doc()
        # Tamper with hash to ensure it's not checked.
        raw["hash_chain"]["top_level_sha256"] = "0" * 64
        multi, result = ingest_from_dict(raw, skip_hash_chain=True)
        assert result.valid
        assert multi.task_count == 25


# ---------------------------------------------------------------------------
# ingest_from_file
# ---------------------------------------------------------------------------


class TestIngestFromFile:
    def test_load_and_ingest(self):
        raw = _make_raw_doc()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
            f.flush()
            path = f.name
        try:
            multi, result = ingest_from_file(path, skip_hash_chain=True)
            assert result.valid
            assert multi.task_count == 25
            assert multi.entries[0].suite == Suite.Mmlu
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ingest_from_file("/nonexistent/path.json")

    def test_v01_file_auto_migrates(self):
        v01 = _make_v01_doc()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(v01, f)
            f.flush()
            path = f.name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                multi, result = ingest_from_file(
                    path, skip_hash_chain=True, allow_v01=True
                )
            assert multi.task_count == 10
            assert multi.correct_count == 7
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dataclass shape compatibility
# ---------------------------------------------------------------------------


class TestDataclassShapes:
    def test_task_result_fields(self):
        tr = TaskResult(task_id="x", suite=Suite.Mmlu, correct=True, score=1.0)
        assert tr.task_id == "x"
        assert tr.suite == Suite.Mmlu
        assert tr.matched_answer is None

    def test_evaluation_report_fields(self):
        er = EvaluationReport(
            suite=Suite.Mmlu,
            task_count=0,
            correct_count=0,
            accuracy=0.0,
            mean_score=0.0,
        )
        assert er.suite == Suite.Mmlu
        assert er.results == []

    def test_suite_report_entry_fields(self):
        prov = DatasetProvenance(
            source="mmlu",
            source_revision="v1",
            split="test",
            content_sha256="a" * 64,
            task_count=0,
        )
        er = EvaluationReport(
            suite=Suite.Mmlu,
            task_count=0,
            correct_count=0,
            accuracy=0.0,
            mean_score=0.0,
        )
        entry = SuiteReportEntry(suite=Suite.Mmlu, provenance=prov, report=er)
        assert entry.provenance.source == "mmlu"

    def test_multi_suite_report_fields(self):
        msr = MultiSuiteReport(
            task_count=0,
            correct_count=0,
            overall_accuracy=0.0,
            mean_suite_accuracy=0.0,
            mean_suite_score=0.0,
        )
        assert msr.entries == []

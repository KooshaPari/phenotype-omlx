"""Unit tests for Eval Interchange Contract v1.0 ingestion."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from .contract import EvalReport, HashChain, ProducerInfo, RunInfo, SuiteResult, Totals
from .loader import load_report, load_report_from_dict
from .validator import (
    ValidationResult,
    validate,
    _canonical_json,
    _compute_top_level_sha256,
)


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
    """Build a raw dict matching the contract schema for testing."""
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
        # compute correct hashes
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


def _parse(doc: dict) -> EvalReport:
    return EvalReport.model_validate(doc)


# ---------------------------------------------------------------------------
# Contract model tests
# ---------------------------------------------------------------------------


class TestContractModels:
    def test_valid_report_roundtrips(self):
        raw = _make_raw_doc()
        report = _parse(raw)
        assert report.contract_version == "1.0"
        assert report.artifact_kind == "EvaluationReport"
        assert report.producer.name == "pheno-harness"
        assert report.suites[0].suite == "mmlu-pro"
        assert report.totals.cells == 25
        assert (
            report.hash_chain.top_level_sha256 == raw["hash_chain"]["top_level_sha256"]
        )

    def test_extra_fields_allowed(self):
        raw = _make_raw_doc()
        raw["matrix"] = {"experimental": True}
        raw["comparator"] = {"delta": 0.05}
        raw["suites"][0]["task_id"] = "abc-123"
        report = _parse(raw)
        assert report.matrix == {"experimental": True}
        assert report.comparator == {"delta": 0.05}

    def test_missing_contract_version_rejected(self):
        raw = _make_raw_doc()
        del raw["contract_version"]
        with pytest.raises(Exception):
            _parse(raw)

    def test_wrong_contract_version_rejected(self):
        raw = _make_raw_doc(contract_version="0.1")
        with pytest.raises(Exception):
            _parse(raw)

    def test_missing_producer_rejected(self):
        raw = _make_raw_doc()
        del raw["producer"]
        with pytest.raises(Exception):
            _parse(raw)

    def test_empty_suites_rejected(self):
        raw = _make_raw_doc(suites=[])
        with pytest.raises(Exception):
            _parse(raw)

    def test_missing_totals_rejected(self):
        raw = _make_raw_doc()
        del raw["totals"]
        with pytest.raises(Exception):
            _parse(raw)

    def test_missing_hash_chain_rejected(self):
        raw = _make_raw_doc()
        del raw["hash_chain"]
        with pytest.raises(Exception):
            _parse(raw)

    def test_invalid_variant_rejected(self):
        raw = _make_raw_doc()
        raw["run"]["variant"] = "invalid"
        with pytest.raises(Exception):
            _parse(raw)

    def test_invalid_judge_mode_rejected(self):
        raw = _make_raw_doc()
        raw["run"]["judge_mode"] = "unknown"
        with pytest.raises(Exception):
            _parse(raw)

    def test_evidence_label_enum(self):
        raw = _make_raw_doc()
        raw["suites"][0]["evidence_label"] = "reported"
        report = _parse(raw)
        assert report.suites[0].evidence_label.value == "reported"

    def test_invalid_evidence_label_rejected(self):
        raw = _make_raw_doc()
        raw["suites"][0]["evidence_label"] = "bogus"
        with pytest.raises(Exception):
            _parse(raw)

    def test_pass_at_1_range_enforced(self):
        raw = _make_raw_doc()
        raw["suites"][0]["pass_at_1"] = 1.5
        with pytest.raises(Exception):
            _parse(raw)

    def test_hash_chain_forbids_extra(self):
        raw = _make_raw_doc()
        raw["hash_chain"]["extra_field"] = "nope"
        with pytest.raises(Exception):
            _parse(raw)


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidator:
    def test_valid_contract_passes(self):
        raw = _make_raw_doc()
        report = _parse(raw)
        result = validate(report, raw)
        assert result.valid
        assert result.errors == []
        # live_verified → no warnings
        assert not any("W-EVIDENCE" in w for w in result.warnings)

    def test_r_version_wrong(self):
        raw = _make_raw_doc(contract_version="1.0")
        report = _parse(raw)
        report.contract_version = "1.1"
        result = validate(report, raw)
        assert not result.valid
        assert any("R-VERSION" in e for e in result.errors)

    def test_r_producer_missing_fields(self):
        raw = _make_raw_doc(producer={"name": "", "version": "", "commit_sha": ""})
        report = _parse(raw)
        result = validate(report, raw)
        assert not result.valid
        assert any("R-PRODUCER" in e for e in result.errors)

    def test_r_suites_empty(self):
        raw = _make_raw_doc(suites=[])
        with pytest.raises(Exception):
            _parse(raw)

    def test_r_totals_missing(self):
        raw = _make_raw_doc()
        report = _parse(raw)
        # simulate totals=None by replacing
        report.totals = None  # type: ignore[assignment]
        result = validate(report, raw)
        assert not result.valid
        assert any("R-TOTALS" in e for e in result.errors)

    def test_w_evidence_non_live(self):
        raw = _make_raw_doc()
        raw["suites"][0]["evidence_label"] = "reported"
        payload = {k: v for k, v in raw.items() if k != "hash_chain"}
        raw["hash_chain"]["top_level_sha256"] = hashlib.sha256(
            _canonical_json(payload)
        ).hexdigest()
        raw["hash_chain"]["task_ids_sorted_sha256"] = hashlib.sha256(b"").hexdigest()
        report = _parse(raw)
        result = validate(report, raw)
        assert result.valid  # warnings don't invalidate
        assert any("W-EVIDENCE" in w for w in result.warnings)
        assert "reported" in result.warnings[0]

    def test_w_evidence_synthetic(self):
        raw = _make_raw_doc()
        raw["suites"][0]["evidence_label"] = "synthetic"
        payload = {k: v for k, v in raw.items() if k != "hash_chain"}
        raw["hash_chain"]["top_level_sha256"] = hashlib.sha256(
            _canonical_json(payload)
        ).hexdigest()
        raw["hash_chain"]["task_ids_sorted_sha256"] = hashlib.sha256(b"").hexdigest()
        report = _parse(raw)
        result = validate(report, raw)
        assert result.valid
        assert any("synthetic" in w for w in result.warnings)

    def test_r_hashchain_top_level_mismatch(self):
        raw = _make_raw_doc()
        report = _parse(raw)
        # tamper with the stored hash
        report.hash_chain.top_level_sha256 = "0" * 64
        result = validate(report, raw)
        assert not result.valid
        assert any("top_level_sha256 mismatch" in e for e in result.errors)

    def test_r_hashchain_task_ids_mismatch(self):
        raw = _make_raw_doc()
        report = _parse(raw)
        report.hash_chain.task_ids_sorted_sha256 = "0" * 64
        result = validate(report, raw)
        assert not result.valid
        assert any("task_ids_sorted_sha256 mismatch" in e for e in result.errors)

    def test_r_hashchain_skipped_when_no_raw_doc(self):
        raw = _make_raw_doc()
        report = _parse(raw)
        result = validate(report, raw_doc=None)
        assert result.valid
        assert any("skipped" in w for w in result.warnings)

    def test_multiple_suites_mixed_evidence(self):
        raw = _make_raw_doc(
            suites=[
                {
                    "suite": "mmlu-pro",
                    "n": 10,
                    "passed": 10,
                    "pass_at_1": 1.0,
                    "evidence_label": "live_verified",
                },
                {
                    "suite": "humaneval",
                    "n": 10,
                    "passed": 8,
                    "pass_at_1": 0.8,
                    "evidence_label": "reported",
                },
            ]
        )
        # Recompute hashes
        payload = {k: v for k, v in raw.items() if k != "hash_chain"}
        raw["hash_chain"]["top_level_sha256"] = hashlib.sha256(
            _canonical_json(payload)
        ).hexdigest()
        raw["hash_chain"]["task_ids_sorted_sha256"] = hashlib.sha256(b"").hexdigest()
        report = _parse(raw)
        result = validate(report, raw)
        assert result.valid
        assert len([w for w in result.warnings if "W-EVIDENCE" in w]) == 1


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestLoader:
    def test_load_from_file(self):
        raw = _make_raw_doc()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
            f.flush()
            path = f.name
        try:
            report, result, doc = load_report(path)
            assert result.valid
            assert report.contract_version == "1.0"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_report("/nonexistent/path.json")

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            f.flush()
            path = f.name
        try:
            with pytest.raises(Exception):
                load_report(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_from_dict(self):
        raw = _make_raw_doc()
        report, result = load_report_from_dict(raw)
        assert result.valid
        assert report.producer.name == "pheno-harness"

    def test_load_dict_validation_fails(self):
        raw = _make_raw_doc()
        report = EvalReport.model_validate(raw)
        report.contract_version = "0.9"
        result = validate(report, raw)
        assert not result.valid
        assert any("R-VERSION" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Hash chain computation tests
# ---------------------------------------------------------------------------


class TestHashChain:
    def test_canonical_json_sorted_keys(self):
        doc = {"z": 1, "a": 2, "m": {"b": 3, "a": 1}}
        canonical = _canonical_json(doc)
        assert b'"a":2' in canonical
        assert b'"z":1' in canonical
        # sorted within nested too
        assert canonical.index(b'"a":1') < canonical.index(b'"b":3')

    def test_canonical_json_no_whitespace(self):
        doc = {"key": [1, 2, 3]}
        canonical = _canonical_json(doc)
        assert b" " not in canonical
        assert b"\n" not in canonical

    def test_top_level_hash_excludes_hash_chain(self):
        raw = _make_raw_doc()
        computed = _compute_top_level_sha256(raw)
        assert computed == raw["hash_chain"]["top_level_sha256"]

    def test_top_level_hash_tamper_detected(self):
        raw = _make_raw_doc()
        original = raw["hash_chain"]["top_level_sha256"]
        raw["contract_version"] = "1.1"
        recomputed = _compute_top_level_sha256(raw)
        assert recomputed != original

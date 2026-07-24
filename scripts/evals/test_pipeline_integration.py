#!/usr/bin/env python3
"""End-to-end integration test: Harbor → cockpit → EvalReport contract.

Validates the full pipeline:
  1. Harbor result.json  →  harbor_to_cockpit converter
  2. cockpit cells        →  interchange ingestion
  3. EvalReport v1.0      →  contract validation

Run:
    python3 -m pytest scripts/evals/test_pipeline_integration.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FIXTURES = REPO_ROOT / "apps" / "bench-cockpit" / "fixtures"
HARBOR_RUN = REPO_ROOT / ".runs" / "harbor-eval-judge-resume" / "2026-07-22__22-39-39"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "evals" / "harbor"))


# ── helpers ──────────────────────────────────────────────────────────


def _cockpit_cell_schema_keys() -> set[str]:
    """Return the required keys every cockpit cell must have."""
    return {
        "suite",
        "task_id",
        "task_title",
        "difficulty",
        "variant",
        "ok",
        "wall_clock_s",
        "tokens_per_second",
        "pass_at_1",
        "gen_ok",
        "partial_credit",
        "judge_score",
        "format_compliance_rate",
        "reply",
        "prompt",
        "expected_answer",
        "scoring_method",
        "total_tokens_in",
        "total_tokens_out",
        "cost_usd",
        "progress_trace",
        "failure_analysis",
        "metadata",
        "created_at",
        "completed_at",
        "model_name",
    }


def _cockpit_summary_schema_keys() -> set[str]:
    """Return the required keys for summary.meta and summary.by_variant."""
    meta = {
        "model",
        "n_suites",
        "n_tasks_per_suite",
        "variants",
        "n_cells",
        "difficulty_mix",
    }
    variant = {"n_cells", "pass_at_1", "mean_wall_clock_s", "ok_count"}
    return meta, variant


def _make_eval_report_from_cockpit(cockpit: dict) -> dict:
    """Synthesize a minimal EvalReport v1.0 from cockpit output for contract testing."""
    cells = cockpit["cells"]
    meta = cockpit["summary"]["meta"]

    # Group cells by suite
    suite_groups: dict[str, list[dict]] = {}
    for cell in cells:
        suite_groups.setdefault(cell["suite"], []).append(cell)

    suites = []
    total_cells = 0
    total_passed = 0
    all_task_ids: list[str] = []

    for suite_name, suite_cells in sorted(suite_groups.items()):
        n = len(suite_cells)
        passed = sum(1 for c in suite_cells if c.get("ok"))
        pass_at_1 = round(sum(c.get("pass_at_1", 0.0) for c in suite_cells) / n, 4)
        task_ids = [c.get("task_id", "") for c in suite_cells]
        all_task_ids.extend(task_ids)

        suites.append(
            {
                "suite": suite_name,
                "n": n,
                "passed": passed,
                "pass_at_1": pass_at_1,
                "evidence_label": "reported",
                "task_ids": task_ids,
            }
        )
        total_cells += n
        total_passed += passed

    overall_pass = round(total_passed / total_cells, 4) if total_cells else 0.0

    # Build hash chain
    task_ids_sorted = sorted(all_task_ids)
    task_hash = hashlib.sha256("\n".join(task_ids_sorted).encode("utf-8")).hexdigest()

    doc = {
        "contract_version": "1.0",
        "artifact_kind": "EvaluationReport",
        "producer": {
            "name": "harbor_to_cockpit",
            "version": "1.0.0",
            "commit_sha": "integration-test",
        },
        "run": {
            "run_id": f"integration-{meta.get('model', 'unknown')}",
            "started_at": "2026-07-22T22:39:39Z",
            "model": meta.get("model", "unknown"),
            "variant": "stock",
            "judge_mode": "deterministic",
        },
        "suites": suites,
        "totals": {
            "cells": total_cells,
            "passed": total_passed,
            "pass_at_1": overall_pass,
        },
        "matrix": None,
        "comparator": None,
        "hash_chain": {
            "top_level_sha256": "",  # computed below
            "task_ids_sorted_sha256": task_hash,
        },
    }

    # Compute top-level hash (everything except hash_chain)
    payload = {k: v for k, v in doc.items() if k != "hash_chain"}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    doc["hash_chain"]["top_level_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    return doc


# ── tests ────────────────────────────────────────────────────────────


class TestHarborToCockpit:
    """Step 1: Harbor result.json → cockpit cells."""

    def test_adapter_runs_on_live_data(self) -> None:
        """The adapter produces cockpit output from the live Harbor run."""
        from harbor_to_cockpit import convert_job

        result = convert_job(HARBOR_RUN)
        assert "summary" in result
        assert "cells" in result
        assert len(result["cells"]) >= 1

    def test_fixture_on_disk_matches_adapter(self) -> None:
        """The committed fixture matches what the adapter produces."""
        from harbor_to_cockpit import convert_job

        fixture_path = FIXTURES / "harbor_oracle_results.json"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"
        fixture = json.loads(fixture_path.read_text())

        live = convert_job(HARBOR_RUN)

        assert len(fixture["cells"]) == len(live["cells"])
        assert fixture["summary"]["meta"]["model"] == live["summary"]["meta"]["model"]
        assert (
            fixture["summary"]["meta"]["n_cells"] == live["summary"]["meta"]["n_cells"]
        )

    def test_cell_schema_completeness(self) -> None:
        """Every cockpit cell has all required keys."""
        from harbor_to_cockpit import convert_job

        result = convert_job(HARBOR_RUN)
        required = _cockpit_cell_schema_keys()
        for cell in result["cells"]:
            missing = required - set(cell.keys())
            assert not missing, f"Cell {cell.get('task_id')} missing keys: {missing}"

    def test_summary_schema_completeness(self) -> None:
        """Summary meta and by_variant have all required keys."""
        from harbor_to_cockpit import convert_job

        result = convert_job(HARBOR_RUN)
        meta_keys, variant_keys = _cockpit_summary_schema_keys()
        meta = result["summary"]["meta"]
        assert meta_keys <= set(meta.keys()), (
            f"Missing meta keys: {meta_keys - set(meta.keys())}"
        )

        for variant_name, bv in result["summary"]["by_variant"].items():
            missing = variant_keys - set(bv.keys())
            assert not missing, f"Variant {variant_name} missing keys: {missing}"


class TestInterchangeIngestion:
    """Step 2: Cockpit output → EvalReport interchange contract."""

    def test_cockpit_to_eval_report_contract(self) -> None:
        """Convert cockpit output to EvalReport v1.0 and validate the contract."""
        from harbor_to_cockpit import convert_job

        cockpit = convert_job(HARBOR_RUN)
        raw_doc = _make_eval_report_from_cockpit(cockpit)

        # Validate Pydantic schema
        from interchange.contract import EvalReport

        report = EvalReport.model_validate(raw_doc)
        assert report.contract_version == "1.0"
        assert report.artifact_kind == "EvaluationReport"
        assert len(report.suites) >= 1
        assert report.totals.cells >= 1

    def test_eval_report_hash_chain_valid(self) -> None:
        """The synthesized EvalReport has a valid hash chain."""
        from harbor_to_cockpit import convert_job

        cockpit = convert_job(HARBOR_RUN)
        raw_doc = _make_eval_report_from_cockpit(cockpit)

        from interchange.contract import EvalReport
        from interchange.validator import validate

        report = EvalReport.model_validate(raw_doc)
        result = validate(report, raw_doc)
        assert result.valid, f"Validation errors: {result.errors}"

    def test_eval_report_validation_catches_tamper(self) -> None:
        """Tampering with the document causes hash chain validation failure."""
        from harbor_to_cockpit import convert_job

        cockpit = convert_job(HARBOR_RUN)
        raw_doc = _make_eval_report_from_cockpit(cockpit)

        # Tamper with a suite value
        raw_doc["suites"][0]["passed"] = 999

        from interchange.contract import EvalReport
        from interchange.validator import validate

        report = EvalReport.model_validate(raw_doc)
        result = validate(report, raw_doc)
        assert not result.valid
        assert any("R-HASHCHAIN" in e for e in result.errors)


class TestFullPipeline:
    """Step 3: End-to-end pipeline summary."""

    def test_full_pipeline_prints_summary(self, capsys: pytest.CaptureFixture) -> None:
        """Run the full pipeline and print a summary."""
        from harbor_to_cockpit import convert_job

        # ── Stage 1: Harbor → cockpit ──
        cockpit = convert_job(HARBOR_RUN)
        n_cells = len(cockpit["cells"])
        model = cockpit["summary"]["meta"]["model"]
        variants = cockpit["summary"]["meta"]["variants"]
        print(
            f"Stage 1 — Harbor → cockpit: {n_cells} cells, model={model}, variants={variants}"
        )

        # ── Stage 2: cockpit → EvalReport ──
        raw_doc = _make_eval_report_from_cockpit(cockpit)

        from interchange.contract import EvalReport

        report = EvalReport.model_validate(raw_doc)
        n_suites = len(report.suites)
        print(
            f"Stage 2 — EvalReport v1.0: {n_suites} suite(s), contract_version={report.contract_version}"
        )

        # ── Stage 3: validate ──
        from interchange.validator import validate

        val_result = validate(report, raw_doc)
        print(
            f"Stage 3 — Validation: valid={val_result.valid} "
            f"errors={len(val_result.errors)} warnings={len(val_result.warnings)}"
        )

        # ── Summary ──
        print(f"\n{'=' * 60}")
        print(f"Pipeline Summary")
        print(f"  Harbor run:      {HARBOR_RUN.name}")
        print(f"  Cells produced:  {n_cells}")
        print(f"  Model:           {model}")
        print(f"  Variants:        {variants}")
        print(f"  Suites:          {n_suites}")
        print(f"  pass_at_1:       {report.totals.pass_at_1}")
        print(f"  Contract valid:  {val_result.valid}")
        print(f"{'=' * 60}")

        # Assertions
        assert n_cells >= 1
        assert report.contract_version == "1.0"
        assert val_result.valid

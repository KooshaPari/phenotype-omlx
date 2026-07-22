"""Offline calibration-evidence contract tests; no model or network access."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "calibration_evidence", ROOT / "scripts" / "calibration_evidence.py"
)
assert SPEC and SPEC.loader
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def _metrics() -> dict[str, float | int]:
    return {
        "token_count": 128,
        "baseline_mean_loss": 1.0,
        "compacted_mean_loss": 1.02,
        "perplexity_ratio": math.exp(0.02),
        "baseline_median_ms_per_token": 2.0,
        "compacted_median_ms_per_token": 1.5,
    }


def _revisions() -> dict[str, str]:
    return {
        "model_revision": "a" * 64,
        "tokenizer_revision": "b" * 64,
        "runtime_revision": "git:" + "c" * 40,
    }


def _qualified_matrix() -> list[dict[str, str]]:
    return [
        {
            "dataset_id": f"release-corpus-{index}",
            "corpus_revision": f"{index:x}" * 64,
            "processed_digest": f"{index + 5:x}" * 64,
        }
        for index in range(1, 6)
    ]


def test_uncalibrated_evidence_is_serializable_with_pinned_revisions_and_metrics() -> None:
    evidence = calibration.CalibrationEvidence.create(
        status="uncalibrated",
        **_revisions(),
        metrics=_metrics(),
        corpus_matrix=[],
    )

    document = evidence.as_dict()

    assert document["schema_version"] == 1
    assert document["status"] == "uncalibrated"
    assert document["release_eligible"] is False
    assert document["runtime_revision"] == "git:" + "c" * 40
    assert document["metrics"] == _metrics()


def test_empty_corpus_matrix_rejects_a_matrix_digest() -> None:
    with pytest.raises(calibration.CalibrationEvidenceError, match="matrix_digest"):
        calibration.CalibrationEvidence.create(
            status="uncalibrated",
            **_revisions(),
            metrics=_metrics(),
            corpus_matrix=[],
            matrix_digest=calibration.corpus_matrix_digest([]),
        )


def test_acceptance_status_requires_a_qualified_five_corpus_matrix() -> None:
    with pytest.raises(calibration.CalibrationEvidenceError, match="qualified corpus matrix"):
        calibration.CalibrationEvidence.create(
            status="acceptance",
            **_revisions(),
            metrics=_metrics(),
            corpus_matrix=[],
        )

    evidence = calibration.CalibrationEvidence.create(
        status="acceptance",
        **_revisions(),
        metrics=_metrics(),
        corpus_matrix=_qualified_matrix(),
        matrix_digest=calibration.corpus_matrix_digest(_qualified_matrix()),
    )
    assert evidence.review_eligible is True
    assert evidence.release_eligible is False


def test_acceptance_status_requires_exactly_five_distinct_corpora_and_matrix_digest() -> None:
    matrix = _qualified_matrix()

    with pytest.raises(calibration.CalibrationEvidenceError, match="exactly five"):
        calibration.CalibrationEvidence.create(
            status="acceptance",
            **_revisions(),
            metrics=_metrics(),
            corpus_matrix=matrix + [
                {
                    "dataset_id": "release-corpus-6",
                    "corpus_revision": "6" * 64,
                    "processed_digest": "b" * 64,
                }
            ],
            matrix_digest="0" * 64,
        )

    with pytest.raises(calibration.CalibrationEvidenceError, match="matrix_digest"):
        calibration.CalibrationEvidence.create(
            status="acceptance",
            **_revisions(),
            metrics=_metrics(),
            corpus_matrix=matrix,
            matrix_digest="0" * 64,
        )


def test_matrix_digest_binds_canonicalized_five_corpus_entries() -> None:
    matrix = list(reversed(_qualified_matrix()))
    digest = calibration.corpus_matrix_digest(matrix)
    evidence = calibration.CalibrationEvidence.create(
        status="acceptance",
        **_revisions(),
        metrics=_metrics(),
        corpus_matrix=matrix,
        matrix_digest=digest,
    )

    document = evidence.as_dict()
    assert document["matrix_digest"] == digest
    assert document["review_eligible"] is True
    assert document["release_eligible"] is False


def test_corpus_matrix_rejects_ids_that_only_differ_by_whitespace() -> None:
    matrix = _qualified_matrix()
    matrix[1]["dataset_id"] = " release-corpus-1 "

    with pytest.raises(calibration.CalibrationEvidenceError, match="dataset_id values must be unique"):
        calibration.CalibrationEvidence.create(
            status="acceptance",
            **_revisions(),
            metrics=_metrics(),
            corpus_matrix=matrix,
            matrix_digest=calibration.corpus_matrix_digest(matrix),
        )


def test_create_rejects_normalized_duplicate_ids_with_a_manually_computed_digest() -> None:
    matrix = _qualified_matrix()
    matrix[1]["dataset_id"] = " release-corpus-1 "
    canonical_entries = sorted(
        (
            {
                "dataset_id": entry["dataset_id"].strip(),
                "corpus_revision": entry["corpus_revision"].lower(),
                "processed_digest": entry["processed_digest"].lower(),
            }
            for entry in matrix
        ),
        key=lambda entry: entry["dataset_id"],
    )
    manually_computed_digest = hashlib.sha256(
        json.dumps(canonical_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(calibration.CalibrationEvidenceError, match="dataset_id values must be unique"):
        calibration.CalibrationEvidence.create(
            status="acceptance",
            **_revisions(),
            metrics=_metrics(),
            corpus_matrix=matrix,
            matrix_digest=manually_computed_digest,
        )


def test_evidence_rejects_mutable_revisions_and_nonfinite_metrics() -> None:
    with pytest.raises(calibration.CalibrationEvidenceError, match="model_revision"):
        calibration.CalibrationEvidence.create(
            status="uncalibrated",
            model_revision="main",
            tokenizer_revision="b" * 64,
            runtime_revision="git:" + "c" * 40,
            metrics=_metrics(),
            corpus_matrix=[],
        )

    metrics = _metrics()
    metrics["perplexity_ratio"] = float("nan")
    with pytest.raises(calibration.CalibrationEvidenceError, match="perplexity_ratio"):
        calibration.CalibrationEvidence.create(
            status="uncalibrated",
            **_revisions(),
            metrics=metrics,
            corpus_matrix=[],
        )


@pytest.mark.parametrize("status", [[], {"acceptance": True}, 1])
def test_evidence_rejects_non_string_status_with_contract_error(status: object) -> None:
    with pytest.raises(calibration.CalibrationEvidenceError, match="status"):
        calibration.CalibrationEvidence.create(
            status=status,
            **_revisions(),
            metrics=_metrics(),
            corpus_matrix=[],
        )


@pytest.mark.parametrize("field", ["baseline_mean_loss", "compacted_mean_loss"])
def test_evidence_rejects_negative_losses(field: str) -> None:
    metrics = _metrics()
    metrics[field] = -0.01

    with pytest.raises(calibration.CalibrationEvidenceError, match=field):
        calibration.CalibrationEvidence.create(
            status="uncalibrated",
            **_revisions(),
            metrics=metrics,
            corpus_matrix=[],
        )


def test_evidence_normalizes_revisions_and_canonicalizes_corpus_order() -> None:
    uppercase_revisions = {
        "model_revision": "A" * 64,
        "tokenizer_revision": "B" * 64,
        "runtime_revision": "git:" + "C" * 40,
    }
    matrix = list(reversed(_qualified_matrix()))
    reversed_matrix = list(reversed(matrix))

    first = calibration.CalibrationEvidence.create(
        status="acceptance",
        **uppercase_revisions,
        metrics=_metrics(),
        corpus_matrix=matrix,
        matrix_digest=calibration.corpus_matrix_digest(matrix),
    )
    second = calibration.CalibrationEvidence.create(
        status="acceptance",
        **uppercase_revisions,
        metrics=_metrics(),
        corpus_matrix=reversed_matrix,
        matrix_digest=calibration.corpus_matrix_digest(reversed_matrix),
    )

    assert first.model_revision == "a" * 64
    assert first.runtime_revision == "git:" + "c" * 40
    assert [entry["dataset_id"] for entry in first.as_dict()["corpus_matrix"]] == [
        f"release-corpus-{index}" for index in range(1, 6)
    ]
    assert json.dumps(first.as_dict(), sort_keys=True, separators=(",", ":")) == json.dumps(
        second.as_dict(), sort_keys=True, separators=(",", ":")
    )

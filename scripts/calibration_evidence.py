#!/usr/bin/env python3
"""Offline, fail-closed calibration evidence schema.

This module validates supplied evidence only.  It never loads a model, reads a
corpus, accesses a network, or publishes a release decision.  In particular,
``acceptance`` is rejected unless a qualified five-corpus matrix accompanies
the immutable model, tokenizer, and runtime revisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence


class CalibrationEvidenceError(ValueError):
    """Raised when an offline calibration record is incomplete or unsafe."""


_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-fA-F]{64}|git:[0-9a-fA-F]{40,64})$")
_DIGEST = re.compile(r"[0-9a-fA-F]{64}$")
_STATUSES = frozenset(("uncalibrated", "acceptance"))
_METRIC_FIELDS = (
    "token_count",
    "baseline_mean_loss",
    "compacted_mean_loss",
    "perplexity_ratio",
    "baseline_median_ms_per_token",
    "compacted_median_ms_per_token",
)


def _require_immutable_revision(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IMMUTABLE_REVISION.fullmatch(value):
        raise CalibrationEvidenceError(f"{name} must be immutable provenance")
    return value.lower()


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise CalibrationEvidenceError(f"{name} must be a 64-hex digest")
    return value.lower()


@dataclass(frozen=True)
class CalibrationMetrics:
    """Finite, same-workload metric fields supplied by a separate evaluator."""

    token_count: int
    baseline_mean_loss: float
    compacted_mean_loss: float
    perplexity_ratio: float
    baseline_median_ms_per_token: float
    compacted_median_ms_per_token: float

    @classmethod
    def create(cls, values: Mapping[str, object]) -> "CalibrationMetrics":
        if not isinstance(values, Mapping) or set(values) != set(_METRIC_FIELDS):
            raise CalibrationEvidenceError("metrics must contain the canonical metric fields")
        token_count = values["token_count"]
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
            raise CalibrationEvidenceError("token_count must be positive")
        finite_values: dict[str, float] = {}
        for field in _METRIC_FIELDS[1:]:
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise CalibrationEvidenceError(f"{field} must be finite")
            if field.endswith("ms_per_token") or field == "perplexity_ratio":
                if value <= 0:
                    raise CalibrationEvidenceError(f"{field} must be positive")
            elif value < 0:
                raise CalibrationEvidenceError(f"{field} must be nonnegative")
            finite_values[field] = float(value)
        return cls(token_count=token_count, **finite_values)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "token_count": self.token_count,
            "baseline_mean_loss": self.baseline_mean_loss,
            "compacted_mean_loss": self.compacted_mean_loss,
            "perplexity_ratio": self.perplexity_ratio,
            "baseline_median_ms_per_token": self.baseline_median_ms_per_token,
            "compacted_median_ms_per_token": self.compacted_median_ms_per_token,
        }


@dataclass(frozen=True)
class CorpusMatrixEntry:
    """One independently pinned, processed release corpus record."""

    dataset_id: str
    corpus_revision: str
    processed_digest: str

    @classmethod
    def create(cls, value: Mapping[str, object]) -> "CorpusMatrixEntry":
        if not isinstance(value, Mapping) or set(value) != {
            "dataset_id",
            "corpus_revision",
            "processed_digest",
        }:
            raise CalibrationEvidenceError("corpus matrix entries have an invalid schema")
        dataset_id = value["dataset_id"]
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise CalibrationEvidenceError("dataset_id must be nonempty")
        return cls(
            dataset_id=dataset_id.strip(),
            corpus_revision=_require_digest("corpus_revision", value["corpus_revision"]),
            processed_digest=_require_digest("processed_digest", value["processed_digest"]),
        )


def _canonical_corpus_matrix(entries: Sequence[CorpusMatrixEntry]) -> bytes:
    """Return the stable, digestable representation of submitted matrix facts."""

    document = [
        {
            "dataset_id": entry.dataset_id,
            "corpus_revision": entry.corpus_revision,
            "processed_digest": entry.processed_digest,
        }
        for entry in sorted(entries, key=lambda entry: entry.dataset_id)
    ]
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def corpus_matrix_digest(corpus_matrix: Sequence[Mapping[str, object]]) -> str:
    """Compute the SHA-256 binding supplied corpus facts without accessing data."""

    if isinstance(corpus_matrix, (str, bytes)) or not isinstance(corpus_matrix, Sequence):
        raise CalibrationEvidenceError("corpus_matrix must be a sequence")
    entries = tuple(CorpusMatrixEntry.create(entry) for entry in corpus_matrix)
    if len({entry.dataset_id for entry in entries}) != len(entries):
        raise CalibrationEvidenceError("corpus_matrix dataset_id values must be unique")
    return hashlib.sha256(_canonical_corpus_matrix(entries)).hexdigest()


@dataclass(frozen=True)
class CalibrationEvidence:
    """Offline calibration evidence; not itself a release authorization."""

    status: str
    model_revision: str
    tokenizer_revision: str
    runtime_revision: str
    metrics: CalibrationMetrics
    corpus_matrix: tuple[CorpusMatrixEntry, ...]
    matrix_digest: str | None

    @classmethod
    def create(
        cls,
        *,
        status: str,
        model_revision: str,
        tokenizer_revision: str,
        runtime_revision: str,
        metrics: Mapping[str, object],
        corpus_matrix: Sequence[Mapping[str, object]],
        matrix_digest: str | None = None,
    ) -> "CalibrationEvidence":
        if not isinstance(status, str) or status not in _STATUSES:
            raise CalibrationEvidenceError("status must be uncalibrated or acceptance")
        if isinstance(corpus_matrix, (str, bytes)) or not isinstance(corpus_matrix, Sequence):
            raise CalibrationEvidenceError("corpus_matrix must be a sequence")
        entries = tuple(sorted(
            (CorpusMatrixEntry.create(entry) for entry in corpus_matrix),
            key=lambda entry: entry.dataset_id,
        ))
        if len({entry.dataset_id for entry in entries}) != len(entries):
            raise CalibrationEvidenceError("corpus_matrix dataset_id values must be unique")
        if entries and len(entries) != 5:
            raise CalibrationEvidenceError("corpus_matrix must contain exactly five distinct corpora")
        if status == "acceptance" and len(entries) != 5:
            raise CalibrationEvidenceError("acceptance requires a qualified corpus matrix of exactly five corpora")
        if not entries:
            if matrix_digest is not None:
                raise CalibrationEvidenceError("matrix_digest requires a corpus_matrix")
        elif matrix_digest is None:
            raise CalibrationEvidenceError("matrix_digest is required for a corpus_matrix")
        else:
            supplied_digest = _require_digest("matrix_digest", matrix_digest)
            expected_digest = hashlib.sha256(_canonical_corpus_matrix(entries)).hexdigest()
            if supplied_digest != expected_digest:
                raise CalibrationEvidenceError("matrix_digest does not bind the supplied corpus_matrix")
            matrix_digest = supplied_digest
        return cls(
            status=status,
            model_revision=_require_immutable_revision("model_revision", model_revision),
            tokenizer_revision=_require_immutable_revision("tokenizer_revision", tokenizer_revision),
            runtime_revision=_require_immutable_revision("runtime_revision", runtime_revision),
            metrics=CalibrationMetrics.create(metrics),
            corpus_matrix=entries,
            matrix_digest=matrix_digest,
        )

    @property
    def review_eligible(self) -> bool:
        """A qualified matrix can enter review, but cannot authorize release."""

        return self.status == "acceptance" and len(self.corpus_matrix) == 5 and self.matrix_digest is not None

    @property
    def release_eligible(self) -> bool:
        """Pure evidence never authorizes a release without external policy approval."""

        return False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "review_eligible": self.review_eligible,
            "release_eligible": self.release_eligible,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "runtime_revision": self.runtime_revision,
            "metrics": self.metrics.as_dict(),
            "matrix_digest": self.matrix_digest,
            "corpus_matrix": [
                {
                    "dataset_id": entry.dataset_id,
                    "corpus_revision": entry.corpus_revision,
                    "processed_digest": entry.processed_digest,
                }
                for entry in self.corpus_matrix
            ],
        }

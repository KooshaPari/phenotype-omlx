#!/usr/bin/env python3
"""Pure orchestration contracts for real-model validation runs.

This module deliberately knows nothing about MLX or model hosting.  A caller
supplies the four cache operations through ``CachePort`` so event provenance is
testable without an inference runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Callable, Protocol, Sequence


class ValidationError(ValueError):
    """Raised when a validation run lacks required evidence."""


@dataclass(frozen=True)
class ValidationManifest:
    """Immutable provenance required before a quality result is meaningful.

    Each revision is either a full 64-hex content digest or ``git:`` followed
    by a 40- to 64-hex immutable commit identifier. Branches, tags, and names
    are intentionally not provenance.
    """

    model_revision: str
    corpus_revision: str
    tokenizer_revision: str

    @classmethod
    def create(
        cls,
        *,
        model_revision: str,
        corpus_revision: str,
        tokenizer_revision: str,
    ) -> "ValidationManifest":
        fields = {
            "model_revision": model_revision,
            "corpus_revision": corpus_revision,
            "tokenizer_revision": tokenizer_revision,
        }
        immutable_revision = re.compile(r"(?:[0-9a-fA-F]{64}|git:[0-9a-fA-F]{40,64})$")
        for name, value in fields.items():
            if not isinstance(value, str) or not immutable_revision.fullmatch(value):
                raise ValidationError(f"{name} must be immutable provenance")
        return cls(**fields)


class CachePort(Protocol):
    """The minimal cache lifecycle used by the compacted validation arm."""

    def prefill(self, cache: object) -> None: ...

    def materialize(self, cache: object) -> None: ...

    def compact(self, cache: object) -> int: ...

    def score(self, cache: object) -> None: ...


@dataclass(frozen=True)
class CompactedArmRun:
    """Evidence that one cache object was compacted before scoring."""

    manifest: ValidationManifest
    cache: object
    cache_identity: int
    saved_bytes: int


@dataclass(frozen=True)
class TeacherForcedScore:
    """Aggregate teacher-forced NLL over one observed token sequence."""

    token_count: int
    total_nll: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count <= 0
        ):
            raise ValidationError("token_count must be positive")
        if not isinstance(self.total_nll, (int, float)) or not math.isfinite(self.total_nll):
            raise ValidationError("total_nll must be finite")

    @property
    def mean_loss(self) -> float:
        return self.total_nll / self.token_count

    @property
    def perplexity(self) -> float:
        try:
            value = math.exp(self.mean_loss)
        except OverflowError as error:
            raise ValidationError("perplexity must be finite") from error
        if not math.isfinite(value):
            raise ValidationError("perplexity must be finite")
        return value


@dataclass(frozen=True)
class TeacherForcedComparison:
    """Baseline-relative loss evidence for the same teacher-forced tokens."""

    baseline: TeacherForcedScore
    compacted: TeacherForcedScore
    compacted_minus_baseline: float
    perplexity_ratio: float


def compare_teacher_forced(
    baseline: TeacherForcedScore,
    compacted: TeacherForcedScore,
) -> TeacherForcedComparison:
    """Compare finite scores only when both arms observed exactly one token count."""

    if not isinstance(baseline, TeacherForcedScore) or not isinstance(compacted, TeacherForcedScore):
        raise ValidationError("teacher-forced scores must be validated")
    if baseline.token_count != compacted.token_count:
        raise ValidationError("token_count must match between baseline and compacted arms")
    difference = compacted.mean_loss - baseline.mean_loss
    try:
        ratio = math.exp(difference)
    except OverflowError as error:
        raise ValidationError("perplexity ratio must be finite") from error
    if not math.isfinite(ratio):
        raise ValidationError("perplexity ratio must be finite")
    return TeacherForcedComparison(
        baseline=baseline,
        compacted=compacted,
        compacted_minus_baseline=difference,
        perplexity_ratio=ratio,
    )


@dataclass(frozen=True)
class SynchronizedSample:
    """One already-synchronized measured or warmup decode repeat."""

    actual_tokens: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.actual_tokens, int)
            or isinstance(self.actual_tokens, bool)
            or self.actual_tokens <= 0
        ):
            raise ValidationError("actual_tokens must be positive")
        if not isinstance(self.elapsed_seconds, (int, float)) or self.elapsed_seconds <= 0:
            raise ValidationError("elapsed_seconds must be positive")
        if not math.isfinite(self.elapsed_seconds):
            raise ValidationError("elapsed_seconds must be finite")

    @property
    def elapsed_per_token(self) -> float:
        return self.elapsed_seconds / self.actual_tokens


@dataclass(frozen=True)
class BenchmarkSummary:
    """Robust timing evidence derived only from observed decode tokens."""

    measured_repeats: int
    actual_tokens: int
    median_elapsed_per_token: float
    p95_elapsed_per_token: float


def summarize_benchmark(
    samples: Sequence[SynchronizedSample], *, warmup_count: int
) -> BenchmarkSummary:
    """Exclude warmups and summarize per-token timing from actual measurements."""

    if not isinstance(warmup_count, int) or isinstance(warmup_count, bool) or warmup_count < 0:
        raise ValidationError("warmup_count must be nonnegative")
    measured = tuple(samples[warmup_count:])
    if not measured:
        raise ValidationError("at least one measured repeat is required")
    if not all(isinstance(sample, SynchronizedSample) for sample in measured):
        raise ValidationError("measured repeats must be synchronized samples")
    elapsed_per_token = sorted(sample.elapsed_per_token for sample in measured)
    p95_index = math.ceil(len(elapsed_per_token) * 0.95) - 1
    return BenchmarkSummary(
        measured_repeats=len(measured),
        actual_tokens=sum(sample.actual_tokens for sample in measured),
        median_elapsed_per_token=statistics.median(elapsed_per_token),
        p95_elapsed_per_token=elapsed_per_token[p95_index],
    )


def _write_fsynced(path: Path, payload: bytes, *, exclusive: bool = False) -> None:
    """Write one payload durably, without replacing an existing exclusive path."""

    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_diagnostic(path: Path, candidate: object) -> Path | None:
    """Best-effort, never-overwriting capture of a rejected candidate."""

    try:
        payload = json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        _write_fsynced(path, payload, exclusive=True)
    except (OSError, TypeError, ValueError):
        return None
    return path


def _require_contained_output(path: Path, approved_output_root: Path, *, label: str) -> Path:
    """Resolve a result path and require its parent to be within the approved root."""

    resolved_root = approved_output_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValidationError("approved output root must be an existing directory")
    resolved_path = path.expanduser().resolve()
    resolved_parent = resolved_path.parent
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise ValidationError(f"{label} must be within the approved output root")
    return resolved_path


def publish_validation_json(
    destination: Path,
    candidate: object,
    *,
    validator: Callable[[object], None],
    approved_output_root: Path,
    diagnostic_path: Path | None = None,
) -> Path:
    """Publish only a validated, reparsed JSON result through atomic replacement.

    The existing canonical result is not opened for writing until candidate
    validation, temporary-file fsync, and validation of the reparsed bytes all
    succeed.  A requested diagnostic capture is exclusive: it cannot replace a
    prior diagnostic or the last-known-good canonical result.
    """

    destination = _require_contained_output(
        Path(destination), Path(approved_output_root), label="destination"
    )
    if diagnostic_path is not None:
        diagnostic_path = _require_contained_output(
            Path(diagnostic_path), Path(approved_output_root), label="diagnostic path"
        )
        if diagnostic_path == destination:
            raise ValidationError("diagnostic path must differ from destination")
    temporary_path: Path | None = None
    replaced = False
    try:
        validator(candidate)
        payload = json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        reparsed = json.loads(temporary_path.read_text(encoding="utf-8"))
        validator(reparsed)
        os.replace(temporary_path, destination)
        temporary_path = None
        replaced = True
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return destination
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if not replaced and diagnostic_path is not None:
            diagnostic = _write_diagnostic(diagnostic_path, candidate)
            if diagnostic is not None:
                return diagnostic
        raise


def run_compacted_arm(
    port: CachePort,
    *,
    cache_factory: Callable[[], object],
    manifest: ValidationManifest | None,
) -> CompactedArmRun:
    """Execute the only valid compacted-cache event sequence.

    The cache is constructed once and passed by identity through prefill,
    materialization, compaction, and teacher-forced scoring.  A zero-byte
    compaction is evidence of no compacted candidate and fails before scoring.
    """

    if not isinstance(manifest, ValidationManifest):
        raise ValidationError("manifest must be validated immutable provenance")
    cache = cache_factory()
    port.prefill(cache)
    port.materialize(cache)
    saved_bytes = port.compact(cache)
    if not isinstance(saved_bytes, int) or isinstance(saved_bytes, bool) or saved_bytes <= 0:
        raise ValidationError("compaction must save a nonzero number of bytes")
    port.score(cache)
    return CompactedArmRun(
        manifest=manifest,
        cache=cache,
        cache_identity=id(cache),
        saved_bytes=saved_bytes,
    )

"""Dependency-injected host boundary for real-model validation evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

try:
    import e2e_validation as validation
except ModuleNotFoundError:
    from scripts import e2e_validation as validation

try:
    from e2e_real_model_probe import LiteProbeResult, run_bounded_probe_process, run_one_token_lite_probe
    from e2e_real_model_workload import BENCHMARK_WORKLOAD_PATH, BenchmarkWorkload, load_benchmark_workload
except ModuleNotFoundError:
    from scripts.e2e_real_model_probe import LiteProbeResult, run_bounded_probe_process, run_one_token_lite_probe
    from scripts.e2e_real_model_workload import BENCHMARK_WORKLOAD_PATH, BenchmarkWorkload, load_benchmark_workload


@dataclass(frozen=True)
class HostCompactedArmRun:
    """Host evidence binding one generated token sequence to one cache object."""

    manifest: validation.ValidationManifest
    cache: object
    cache_identity: int
    generated_tokens: tuple[int, ...]
    saved_bytes: int


@dataclass(frozen=True)
class DecodeObservation:
    """One synchronized generation measurement using observed output tokens."""

    text: str
    actual_tokens: int
    elapsed_seconds: float


class TeacherForcedScorer(Protocol):
    """Runtime adapter that produces same-workload baseline/compacted evidence."""

    def __call__(
        self,
        *,
        model: object,
        tokenizer: object,
        workload: "BenchmarkWorkload",
        mx: object,
    ) -> validation.TeacherForcedComparison: ...


class TeacherForcedCapabilityError(RuntimeError):
    """Raised before benchmarks when the installed cache cannot be scored safely."""


def validation_manifest_from_environment(
    environ: Mapping[str, str] | None = None,
) -> validation.ValidationManifest:
    """Construct required immutable provenance before touching model dependencies."""

    values = os.environ if environ is None else environ
    return validation.ValidationManifest.create(
        model_revision=values.get("PHENO_MODEL_REVISION", ""),
        corpus_revision=values.get("PHENO_CORPUS_REVISION", ""),
        tokenizer_revision=values.get("PHENO_TOKENIZER_REVISION", ""),
    )


def validation_manifest_for_workload(
    environ: Mapping[str, str] | None = None,
    *,
    workload: BenchmarkWorkload,
) -> validation.ValidationManifest:
    """Bind a local workload digest to the corpus-revision evidence field."""

    values = dict(os.environ if environ is None else environ)
    supplied_revision = values.get("PHENO_CORPUS_REVISION")
    if supplied_revision is not None and supplied_revision != workload.revision:
        raise validation.ValidationError("corpus_revision must match checked-in benchmark workload")
    values["PHENO_CORPUS_REVISION"] = workload.revision
    return validation_manifest_from_environment(values)


def run_host_validation(
    *,
    environ: Mapping[str, str] | None,
    load_model: Callable[[], object],
    run_arm: Callable[[object], object],
) -> object:
    """Validate provenance before model loading, then delegate to a host arm."""

    validation_manifest_from_environment(environ)
    return run_arm(load_model())


def observe_synchronized_decode(
    *,
    generate: Callable[[], str],
    tokenize: Callable[[str], Sequence[object]],
    synchronize: Callable[[], None],
    clock: Callable[[], float],
) -> DecodeObservation:
    """Time generation only after its device work is synchronized."""

    started = clock()
    text = generate()
    synchronize()
    elapsed_seconds = clock() - started
    actual_tokens = len(tuple(tokenize(text)))
    return DecodeObservation(
        text=text,
        actual_tokens=actual_tokens,
        elapsed_seconds=elapsed_seconds,
    )


def compare_teacher_forced_nll(
    *,
    token_count: int,
    baseline_total_nll: float,
    compacted_total_nll: float,
) -> validation.TeacherForcedComparison:
    """Create finite same-token quality evidence for a compacted arm."""

    baseline = validation.TeacherForcedScore(
        token_count=token_count,
        total_nll=baseline_total_nll,
    )
    compacted = validation.TeacherForcedScore(
        token_count=token_count,
        total_nll=compacted_total_nll,
    )
    return validation.compare_teacher_forced(baseline, compacted)


def run_teacher_forced_scorer(
    *,
    scorer: TeacherForcedScorer,
    model: object,
    tokenizer: object,
    workload: BenchmarkWorkload,
    mx: object,
) -> validation.TeacherForcedComparison:
    """Invoke a runtime scorer while preserving the fail-closed policy boundary."""

    comparison = scorer(model=model, tokenizer=tokenizer, workload=workload, mx=mx)
    if not isinstance(comparison, validation.TeacherForcedComparison):
        raise validation.ValidationError("teacher-forced scorer must return a valid comparison")
    return comparison


def default_teacher_forced_scorer_factory(cache_type: type[object]) -> TeacherForcedScorer:
    """Reject cache implementations lacking the verified compact operation.

    The installed ``compact_turbo_cache`` helper operates on
    ``TurboKVCacheLite`` only. A ``TurboKVCache`` instance has no compatible
    ``compact()`` method, so scoring it as "compacted" would fabricate
    evidence. The Lite continuation-scoring adapter must be separately
    validated against a real model before this factory can return one.
    """

    if not callable(getattr(cache_type, "compact", None)):
        raise TeacherForcedCapabilityError(
            "installed TurboKVCache lacks compact(); compact_turbo_cache supports "
            "TurboKVCacheLite only, so no teacher-forced scorer is available"
        )
    raise TeacherForcedCapabilityError(
        "TurboKVCacheLite continuation-scoring semantics require real-model validation "
        "before a default teacher-forced scorer can be enabled"
    )


def run_lite_teacher_forced_lifecycle(
    *,
    workload: BenchmarkWorkload,
    stock_cache_factory: Callable[[], object],
    lite_cache_factory: Callable[[], object],
    prefill: Callable[[object, str], float],
    compact: Callable[[object], int],
    score_continuation: Callable[[object, str], float],
    token_count: int,
) -> validation.TeacherForcedComparison:
    """Score the same frozen continuation after independent stock/Lite prefill.

    This is intentionally a callback seam: model-specific logits and cache
    operations stay out of policy until a real model validates them.
    """

    stock_cache = stock_cache_factory()
    baseline_total_nll = prefill(stock_cache, workload.prompt)
    baseline_total_nll += score_continuation(stock_cache, workload.teacher_forced_continuation)
    lite_cache = lite_cache_factory()
    compacted_total_nll = prefill(lite_cache, workload.prompt)
    saved_bytes = compact(lite_cache)
    if not isinstance(saved_bytes, int) or isinstance(saved_bytes, bool) or saved_bytes <= 0:
        raise validation.ValidationError("Lite cache compaction must save nonzero bytes")
    compacted_total_nll += score_continuation(lite_cache, workload.teacher_forced_continuation)
    return compare_teacher_forced_nll(
        token_count=token_count,
        baseline_total_nll=baseline_total_nll,
        compacted_total_nll=compacted_total_nll,
    )


def benchmark_repeated_arm(
    *,
    run_once: Callable[[], Mapping[str, object]],
    repeats: int,
    warmup_count: int,
) -> dict[str, object]:
    """Run a synchronized arm repeatedly and publish warmup-free median timing."""

    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise validation.ValidationError("repeats must be positive")
    runs = [dict(run_once()) for _ in range(repeats)]
    samples = [
        validation.SynchronizedSample(
            actual_tokens=run["actual_tokens"],
            elapsed_seconds=run["elapsed_s"],
        )
        for run in runs
    ]
    summary = validation.summarize_benchmark(samples, warmup_count=warmup_count)
    result = dict(runs[-1])
    result["tok_per_s"] = 1 / summary.median_elapsed_per_token
    result["benchmark"] = {
        "warmup_count": warmup_count,
        "measured_repeats": summary.measured_repeats,
        "actual_tokens": summary.actual_tokens,
        "median_elapsed_per_token": summary.median_elapsed_per_token,
        "p95_elapsed_per_token": summary.p95_elapsed_per_token,
    }
    return result


def run_host_compacted_arm(
    *,
    manifest: validation.ValidationManifest,
    cache_factory: Callable[[], object],
    generate: Callable[[object], Sequence[int]],
    materialize: Callable[[object, tuple[int, ...]], None],
    compact: Callable[[object], int],
    score: Callable[[object, tuple[int, ...]], None],
) -> HostCompactedArmRun:
    """Generate, compact, and score with precisely one cache and token sequence."""

    if not isinstance(manifest, validation.ValidationManifest):
        raise validation.ValidationError("manifest must be validated immutable provenance")
    cache = cache_factory()
    generated_tokens = tuple(generate(cache))
    if not generated_tokens:
        raise validation.ValidationError("generation must produce observed tokens")
    materialize(cache, generated_tokens)
    saved_bytes = compact(cache)
    if not isinstance(saved_bytes, int) or isinstance(saved_bytes, bool) or saved_bytes <= 0:
        raise validation.ValidationError("compaction must save a nonzero number of bytes")
    score(cache, generated_tokens)
    return HostCompactedArmRun(
        manifest=manifest,
        cache=cache,
        cache_identity=id(cache),
        generated_tokens=generated_tokens,
        saved_bytes=saved_bytes,
    )


def collect_synchronized_decode_benchmark(
    *,
    repeats: int,
    warmup_count: int,
    decode_once: Callable[[], tuple[int, float]],
    synchronize: Callable[[], None],
) -> validation.BenchmarkSummary:
    """Collect synchronized repeats using decoded, not requested, token counts."""

    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise validation.ValidationError("repeats must be positive")
    samples: list[validation.SynchronizedSample] = []
    for _ in range(repeats):
        actual_tokens, elapsed_seconds = decode_once()
        synchronize()
        samples.append(
            validation.SynchronizedSample(
                actual_tokens=actual_tokens,
                elapsed_seconds=elapsed_seconds,
            )
        )
    return validation.summarize_benchmark(samples, warmup_count=warmup_count)


def _validate_host_evidence(candidate: object) -> None:
    """Require immutable revisions in every published host evidence record."""

    if not isinstance(candidate, Mapping):
        raise validation.ValidationError("host evidence must be a mapping")
    validation.ValidationManifest.create(
        model_revision=candidate.get("model_revision", ""),
        corpus_revision=candidate.get("corpus_revision", ""),
        tokenizer_revision=candidate.get("tokenizer_revision", ""),
    )


def publish_host_validation_evidence(
    *,
    destination: Path,
    candidate: object,
    approved_output_root: Path,
) -> Path:
    """Publish host evidence only through the atomic pure-policy publisher."""

    return validation.publish_validation_json(
        destination,
        candidate,
        validator=_validate_host_evidence,
        approved_output_root=approved_output_root,
    )

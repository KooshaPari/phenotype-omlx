"""Contract tests for deterministic real-model validation orchestration."""

from __future__ import annotations

import importlib.util
import math
import os
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "e2e_validation", ROOT / "scripts" / "e2e_validation.py"
)
assert SPEC and SPEC.loader
validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation
SPEC.loader.exec_module(validation)


def _manifest() -> object:
    return validation.ValidationManifest.create(
        model_revision="a" * 64,
        corpus_revision="b" * 64,
        tokenizer_revision="c" * 64,
    )


def test_manifest_rejects_missing_immutable_model_or_corpus_provenance() -> None:
    with pytest.raises(validation.ValidationError, match="model_revision"):
        validation.ValidationManifest.create(
            model_revision="",
            corpus_revision="b" * 64,
            tokenizer_revision="c" * 64,
        )

    with pytest.raises(validation.ValidationError, match="corpus_revision"):
        validation.ValidationManifest.create(
            model_revision="a" * 64,
            corpus_revision="",
            tokenizer_revision="c" * 64,
        )


@pytest.mark.parametrize("field", ["model_revision", "corpus_revision", "tokenizer_revision"])
@pytest.mark.parametrize("mutable_ref", ["main", "latest", "Qwen2.5-0.5B"])
def test_manifest_rejects_mutable_or_bare_provenance(field: str, mutable_ref: str) -> None:
    values = {
        "model_revision": "a" * 64,
        "corpus_revision": "b" * 64,
        "tokenizer_revision": "c" * 64,
    }
    values[field] = mutable_ref

    with pytest.raises(validation.ValidationError, match=field):
        validation.ValidationManifest.create(**values)

def test_same_cache_sequence_has_one_identity_and_exact_event_order() -> None:
    events: list[tuple[str, int]] = []

    class FakeCachePort:
        def prefill(self, cache: object) -> None:
            events.append(("prefill", id(cache)))

        def materialize(self, cache: object) -> None:
            events.append(("materialize", id(cache)))

        def compact(self, cache: object) -> int:
            events.append(("compact", id(cache)))
            return 512

        def score(self, cache: object) -> None:
            events.append(("score", id(cache)))

    run = validation.run_compacted_arm(
        FakeCachePort(), cache_factory=object, manifest=_manifest()
    )

    assert run.saved_bytes == 512
    assert run.cache_identity == id(run.cache)
    assert events == [
        ("prefill", run.cache_identity),
        ("materialize", run.cache_identity),
        ("compact", run.cache_identity),
        ("score", run.cache_identity),
    ]


def test_same_cache_sequence_rejects_zero_compaction() -> None:
    class NoCompactionPort:
        def prefill(self, cache: object) -> None:
            pass

        def materialize(self, cache: object) -> None:
            pass

        def compact(self, cache: object) -> int:
            return 0

        def score(self, cache: object) -> None:
            raise AssertionError("score must not run after failed compaction")

    with pytest.raises(validation.ValidationError, match="nonzero"):
        validation.run_compacted_arm(
            NoCompactionPort(), cache_factory=object, manifest=_manifest()
        )


def test_same_cache_sequence_rejects_missing_manifest() -> None:
    class ValidPort:
        def prefill(self, cache: object) -> None:
            pass

        def materialize(self, cache: object) -> None:
            pass

        def compact(self, cache: object) -> int:
            return 1

        def score(self, cache: object) -> None:
            pass

    with pytest.raises(validation.ValidationError, match="manifest"):
        validation.run_compacted_arm(ValidPort(), cache_factory=object, manifest=None)


def test_teacher_forced_scores_require_matching_finite_token_evidence() -> None:
    baseline = validation.TeacherForcedScore(token_count=4, total_nll=2.0)
    compacted = validation.TeacherForcedScore(token_count=4, total_nll=2.4)

    comparison = validation.compare_teacher_forced(baseline, compacted)

    assert comparison.baseline.mean_loss == pytest.approx(0.5)
    assert comparison.compacted.mean_loss == pytest.approx(0.6)
    assert comparison.baseline.perplexity == pytest.approx(math.exp(0.5))
    assert comparison.compacted.perplexity == pytest.approx(math.exp(0.6))
    assert comparison.compacted_minus_baseline == pytest.approx(0.1)
    assert comparison.perplexity_ratio == pytest.approx(math.exp(0.1))

    with pytest.raises(validation.ValidationError, match="token_count"):
        validation.compare_teacher_forced(
            baseline, validation.TeacherForcedScore(token_count=3, total_nll=2.4)
        )
    with pytest.raises(validation.ValidationError, match="finite"):
        validation.compare_teacher_forced(
            baseline, validation.TeacherForcedScore(token_count=4, total_nll=float("nan"))
        )
    with pytest.raises(validation.ValidationError, match="positive"):
        validation.TeacherForcedScore(token_count=0, total_nll=0.0)


def test_benchmark_summary_excludes_warmups_and_uses_actual_tokens() -> None:
    samples = [
        validation.SynchronizedSample(actual_tokens=100, elapsed_seconds=100.0),
        validation.SynchronizedSample(actual_tokens=2, elapsed_seconds=0.2),
        validation.SynchronizedSample(actual_tokens=10, elapsed_seconds=2.0),
    ]

    summary = validation.summarize_benchmark(samples, warmup_count=1)

    assert summary.measured_repeats == 2
    assert summary.actual_tokens == 12
    assert summary.median_elapsed_per_token == pytest.approx(0.15)
    assert summary.p95_elapsed_per_token == pytest.approx(0.2)

    with pytest.raises(validation.ValidationError, match="warmup_count"):
        validation.summarize_benchmark(samples, warmup_count=-1)
    with pytest.raises(validation.ValidationError, match="measured"):
        validation.summarize_benchmark(samples, warmup_count=len(samples))
    with pytest.raises(validation.ValidationError, match="positive"):
        validation.SynchronizedSample(actual_tokens=0, elapsed_seconds=1.0)


def test_atomic_json_publisher_replaces_canonical_only_after_double_validation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"state":"last-good"}', encoding="utf-8")
    calls: list[dict[str, object]] = []

    def validate(candidate: object) -> None:
        assert isinstance(candidate, dict)
        calls.append(candidate)
        if candidate.get("state") != "valid":
            raise validation.ValidationError("candidate is invalid")

    published = validation.publish_validation_json(
        destination,
        {"state": "valid", "metric": 1.25},
        validator=validate,
        approved_output_root=tmp_path,
    )

    assert published == destination
    assert destination.read_text(encoding="utf-8") == '{"metric":1.25,"state":"valid"}'
    assert calls == [
        {"state": "valid", "metric": 1.25},
        {"metric": 1.25, "state": "valid"},
    ]
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_json_publisher_preserves_last_good_file_on_invalid_candidate(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.json"
    diagnostic = tmp_path / "invalid-candidate.json"
    last_good = '{"state":"last-good"}'
    destination.write_text(last_good, encoding="utf-8")

    def validate(candidate: object) -> None:
        assert isinstance(candidate, dict)
        if candidate.get("state") != "valid":
            raise validation.ValidationError("candidate is invalid")

    diagnostic_result = validation.publish_validation_json(
        destination,
        {"state": "invalid"},
        validator=validate,
        approved_output_root=tmp_path,
        diagnostic_path=diagnostic,
    )

    assert diagnostic_result == diagnostic
    assert destination.read_text(encoding="utf-8") == last_good
    assert diagnostic.read_text(encoding="utf-8") == '{"state":"invalid"}'


def test_atomic_json_publisher_rejects_destination_outside_approved_root(tmp_path: Path) -> None:
    approved_root = tmp_path / "approved"
    approved_root.mkdir()

    with pytest.raises(validation.ValidationError, match="approved output root"):
        validation.publish_validation_json(
            tmp_path / "outside.json",
            {"state": "valid"},
            validator=lambda _: None,
            approved_output_root=approved_root,
        )


def test_atomic_json_publisher_rejects_diagnostic_collision_with_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.json"

    with pytest.raises(validation.ValidationError, match="diagnostic path"):
        validation.publish_validation_json(
            destination,
            {"state": "invalid"},
            validator=lambda _: (_ for _ in ()).throw(validation.ValidationError("invalid")),
            approved_output_root=tmp_path,
            diagnostic_path=destination,
        )


def test_atomic_json_publisher_rejects_nonstandard_nan_json(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        validation.publish_validation_json(
            destination,
            {"state": "valid", "metric": float("nan")},
            validator=lambda _: None,
            approved_output_root=tmp_path,
        )

    assert not destination.exists()


def test_atomic_json_publisher_propagates_post_replace_directory_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed durability barrier is not a rejected candidate.

    Once replacement has occurred, publishing a diagnostic and returning it
    would incorrectly imply the old canonical result survived.  The failure
    must reach the caller, while the caller can observe that replacement did
    happen and decide how to recover.
    """
    destination = tmp_path / "result.json"
    diagnostic = tmp_path / "invalid-candidate.json"
    destination.write_text('{"state":"last-good"}', encoding="utf-8")
    original_fsync = validation.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(validation.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        validation.publish_validation_json(
            destination,
            {"state": "valid"},
            validator=lambda _: None,
            approved_output_root=tmp_path,
            diagnostic_path=diagnostic,
        )

    assert destination.read_text(encoding="utf-8") == '{"state":"valid"}'
    assert not diagnostic.exists()

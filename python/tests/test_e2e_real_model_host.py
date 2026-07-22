"""Host-boundary contracts for the real-model evidence harness.

These tests deliberately substitute every MLX/model operation.  They define
the host bridge that ``scripts/e2e_real_model.py`` must provide while keeping
the pure evidence policies in ``scripts/e2e_validation.py`` independently
testable.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validation = _load_module("e2e_validation", ROOT / "scripts" / "e2e_validation.py")
host = _load_module("e2e_real_model_under_test", ROOT / "scripts" / "e2e_real_model.py")


def _revisions() -> dict[str, str]:
    return {
        "PHENO_MODEL_REVISION": "a" * 64,
        "PHENO_CORPUS_REVISION": "b" * 64,
        "PHENO_TOKENIZER_REVISION": "c" * 64,
    }


def test_manifest_is_immutable_and_rejected_before_model_import() -> None:
    calls: list[str] = []

    def must_not_load_model() -> object:
        calls.append("load-model")
        raise AssertionError("model import/load must follow provenance validation")

    with pytest.raises(validation.ValidationError, match="model_revision"):
        host.run_host_validation(
            environ={},
            load_model=must_not_load_model,
            run_arm=lambda _model: None,
        )

    assert calls == []

    manifest = host.validation_manifest_from_environment(_revisions())
    assert manifest.model_revision == "a" * 64
    assert manifest.corpus_revision == "b" * 64
    assert manifest.tokenizer_revision == "c" * 64


def test_host_compacted_arm_uses_one_cache_and_the_exact_generated_tokens() -> None:
    cache = object()
    events: list[tuple[str, int, tuple[int, ...] | None]] = []
    manifest = validation.ValidationManifest.create(
        model_revision="a" * 64,
        corpus_revision="b" * 64,
        tokenizer_revision="c" * 64,
    )

    def generate(received_cache: object) -> list[int]:
        events.append(("generate", id(received_cache), None))
        return [7, 11, 13]

    def materialize(received_cache: object, tokens: tuple[int, ...]) -> None:
        events.append(("materialize", id(received_cache), tokens))

    def compact(received_cache: object) -> int:
        events.append(("compact", id(received_cache), None))
        return 256

    def score(received_cache: object, tokens: tuple[int, ...]) -> None:
        events.append(("score", id(received_cache), tokens))

    run = host.run_host_compacted_arm(
        manifest=manifest,
        cache_factory=lambda: cache,
        generate=generate,
        materialize=materialize,
        compact=compact,
        score=score,
    )

    assert run.cache_identity == id(cache)
    assert run.generated_tokens == (7, 11, 13)
    assert events == [
        ("generate", id(cache), None),
        ("materialize", id(cache), (7, 11, 13)),
        ("compact", id(cache), None),
        ("score", id(cache), (7, 11, 13)),
    ]


def test_decode_benchmark_excludes_warmups_and_counts_observed_tokens() -> None:
    samples = iter([(999, 99.0), (2, 0.4), (3, 0.9)])
    synchronizations: list[str] = []

    summary = host.collect_synchronized_decode_benchmark(
        repeats=3,
        warmup_count=1,
        decode_once=lambda: next(samples),
        synchronize=lambda: synchronizations.append("sync"),
    )

    assert synchronizations == ["sync", "sync", "sync"]
    assert summary.measured_repeats == 2
    assert summary.actual_tokens == 5
    assert summary.median_elapsed_per_token == pytest.approx(0.25)


def test_observed_decode_synchronizes_before_timing_end_and_ignores_requested_budget() -> None:
    events: list[str] = []
    clock_values = iter([10.0, 10.5])

    observation = host.observe_synchronized_decode(
        generate=lambda: events.append("generate") or "early stop",
        tokenize=lambda text: [text, "token", "count"],
        synchronize=lambda: events.append("synchronize"),
        clock=lambda: events.append("clock") or next(clock_values),
    )

    assert events == ["clock", "generate", "synchronize", "clock"]
    assert observation.text == "early stop"
    assert observation.actual_tokens == 3
    assert observation.elapsed_seconds == pytest.approx(0.5)


def test_teacher_forced_quality_compares_finite_same_token_evidence() -> None:
    comparison = host.compare_teacher_forced_nll(
        token_count=4,
        baseline_total_nll=4.0,
        compacted_total_nll=4.4,
    )

    assert comparison.baseline.perplexity == pytest.approx(pytest.importorskip("math").e)
    assert comparison.compacted_minus_baseline == pytest.approx(0.1)
    assert comparison.perplexity_ratio == pytest.approx(pytest.importorskip("math").exp(0.1))

    with pytest.raises(validation.ValidationError, match="finite"):
        host.compare_teacher_forced_nll(
            token_count=4,
            baseline_total_nll=float("nan"),
            compacted_total_nll=4.4,
        )


def test_injected_teacher_forced_scorer_requires_a_valid_comparison() -> None:
    workload = host.load_benchmark_workload()
    comparison = host.run_teacher_forced_scorer(
        scorer=lambda **_kwargs: host.compare_teacher_forced_nll(
            token_count=3,
            baseline_total_nll=3.0,
            compacted_total_nll=3.3,
        ),
        model=object(),
        tokenizer=object(),
        workload=workload,
        mx=object(),
    )
    assert comparison.baseline.token_count == 3

    with pytest.raises(validation.ValidationError, match="comparison"):
        host.run_teacher_forced_scorer(
            scorer=lambda **_kwargs: None,
            model=object(),
            tokenizer=object(),
            workload=workload,
            mx=object(),
        )


def test_default_teacher_forced_factory_refuses_cache_without_compact_capability() -> None:
    class UnsupportedTurboCache:
        pass

    with pytest.raises(host.TeacherForcedCapabilityError, match="TurboKVCacheLite"):
        host.default_teacher_forced_scorer_factory(UnsupportedTurboCache)


def test_lite_teacher_forced_lifecycle_prefills_then_compacts_before_same_targets() -> None:
    events: list[tuple[str, object]] = []
    workload = host.load_benchmark_workload()

    comparison = host.run_lite_teacher_forced_lifecycle(
        workload=workload,
        stock_cache_factory=lambda: "stock",
        lite_cache_factory=lambda: "lite",
        prefill=lambda cache, prompt: events.append(("prefill", cache)) or 1.0,
        compact=lambda cache: events.append(("compact", cache)) or 64,
        score_continuation=lambda cache, continuation: events.append(("score", cache)) or (
            1.2 if cache == "lite" else 1.1
        ),
        token_count=3,
    )

    assert events == [("prefill", "stock"), ("score", "stock"), ("prefill", "lite"), ("compact", "lite"), ("score", "lite")]
    assert comparison.compacted_minus_baseline == pytest.approx(0.1 / 3)


def test_one_token_lite_probe_uses_separate_caches_and_never_publishes() -> None:
    events: list[tuple[str, object, object]] = []
    workload = host.load_benchmark_workload()

    def bounded(phase: str, operation, timeout_seconds: float):
        events.append(("bounded", phase, timeout_seconds))
        return operation()

    result = host.run_one_token_lite_probe(
        workload=workload,
        stock_cache_factory=lambda: "stock",
        lite_cache_factory=lambda: "lite",
        prefill=lambda cache, prompt: events.append(("prefill", cache, prompt)) or object(),
        compact=lambda cache: events.append(("compact", cache, "")) or 64,
        score_one_token=lambda cache, token: events.append(("score", cache, token)) or object(),
        first_token_id=lambda _continuation: [42],
        logits_are_finite=lambda _logits: True,
        bounded_execute=bounded,
        timeout_seconds=5.0,
    )

    assert result.saved_bytes == 64
    assert result.baseline_logits_finite and result.compacted_logits_finite
    assert ("score", "stock", 42) in events
    assert ("score", "lite", 42) in events
    assert [event[1] for event in events if event[0] == "bounded"] == [
        "stock-prefill", "stock-score", "lite-prefill", "lite-compact", "lite-score"
    ]


@pytest.mark.parametrize("tokens", [[], [1, 2], ["1"]])
def test_one_token_lite_probe_rejects_non_single_token_ids(tokens: list[object]) -> None:
    workload = host.load_benchmark_workload()

    with pytest.raises(validation.ValidationError, match="exactly one integer token"):
        host.run_one_token_lite_probe(
            workload=workload,
            stock_cache_factory=object,
            lite_cache_factory=object,
            prefill=lambda *_args: object(),
            compact=lambda _cache: 1,
            score_one_token=lambda *_args: object(),
            first_token_id=lambda _continuation: tokens,
            logits_are_finite=lambda _logits: True,
            bounded_execute=lambda _phase, operation, _timeout: operation(),
            timeout_seconds=5.0,
        )


def _probe_revision_binding() -> dict[str, str]:
    return {"model_revision": "a" * 64, "workload_revision": "b" * 64}


def test_bounded_probe_process_sends_only_serialized_inputs_and_parses_json(tmp_path: Path) -> None:
    revision = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
    model_path = "/Users/kooshapari/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/" + revision

    result = host.run_bounded_probe_process(
        command=[sys.executable, str(ROOT / "scripts" / "lite_probe_child.py")],
        request={
            "model_path": model_path,
            "model_revision": "git:" + revision,
            "tokenizer_revision": "b" * 64,
            "workload_path": str(ROOT / "scripts" / "workloads" / "fibonacci-teacher-forced-v1.json"),
            "workload_revision": "0edd5cab55ad65d7a4e471df507e2a4426a492f19bb47a4a00d000492a5c3e66",
        },
        timeout_seconds=2.0,
    )

    assert result["status"] == "capability_pending"


@pytest.mark.parametrize(
    "response",
    ["[]", '{"status":"capability_pending","publication":true}', '{"status":"other","publication":false}'],
)
def test_bounded_probe_process_rejects_invalid_capability_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    child = tmp_path / "lite_probe_child.py"
    child.write_text(f"print({response!r})\n", encoding="utf-8")
    monkeypatch.setitem(host.run_bounded_probe_process.__globals__, "__file__", str(tmp_path / "e2e_real_model_host.py"))

    with pytest.raises(validation.ValidationError):
        host.run_bounded_probe_process(
            command=[sys.executable, str(child)], request=_probe_revision_binding(), timeout_seconds=1.0
        )


@pytest.mark.parametrize("timeout", [0, -1, True, 31])
def test_bounded_probe_process_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(validation.ValidationError):
        host.run_bounded_probe_process(command=[sys.executable, "ignored"], request={}, timeout_seconds=timeout)


def test_bounded_probe_process_requires_approved_python_interpreter() -> None:
    with pytest.raises(validation.ValidationError, match="approved Python"):
        host.run_bounded_probe_process(
            command=["/bin/sh", str(ROOT / "scripts" / "lite_probe_child.py")],
            request=_probe_revision_binding(),
            timeout_seconds=1.0,
        )


def test_bounded_probe_process_rejects_response_not_bound_to_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "lite_probe_child.py"
    child.write_text(
        "import json\nprint(json.dumps({'status':'capability_pending','publication':False,"
        "'model_revision':'wrong','workload_revision':'wrong'}))\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(host.run_bounded_probe_process.__globals__, "__file__", str(tmp_path / "e2e_real_model_host.py"))
    with pytest.raises(validation.ValidationError, match="does not bind"):
        host.run_bounded_probe_process(
            command=[sys.executable, str(child)],
            request=_probe_revision_binding(),
            timeout_seconds=1.0,
        )


def test_bounded_probe_process_kills_new_process_group_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 4242
        returncode = None

        def communicate(self, _payload=None, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(["python"], timeout)
            return "", ""

    captured: dict[str, object] = {}
    killed: list[tuple[int, int]] = []

    def fake_popen(*_args, **kwargs):
        captured.update(kwargs)
        return Process()

    runner_globals = host.run_bounded_probe_process.__globals__
    monkeypatch.setattr(runner_globals["subprocess"], "Popen", fake_popen)
    monkeypatch.setattr(runner_globals["os"], "killpg", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(TimeoutError, match="exceeded"):
        host.run_bounded_probe_process(
            command=[sys.executable, str(ROOT / "scripts" / "lite_probe_child.py")],
            request=_probe_revision_binding(),
            timeout_seconds=0.01,
        )

    assert captured["start_new_session"] is True
    assert captured["env"] == {"PATH": os.defpath, "PYTHONUNBUFFERED": "1"}
    assert killed == [(4242, runner_globals["signal"].SIGKILL)]


def test_bounded_probe_process_drops_parent_environment_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "lite_probe_child.py"
    child.write_text(
        "import json, os\nassert 'PROBE_PARENT_SENTINEL' not in os.environ\n"
        "print(json.dumps({'status':'capability_pending','publication':False,"
        "'model_revision':'" + "a" * 64 + "','workload_revision':'" + "b" * 64 + "'}))\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(host.run_bounded_probe_process.__globals__, "__file__", str(tmp_path / "e2e_real_model_host.py"))
    monkeypatch.setenv("PROBE_PARENT_SENTINEL", "must-not-cross-boundary")
    result = host.run_bounded_probe_process(
        command=[sys.executable, str(child)], request=_probe_revision_binding(), timeout_seconds=1.0
    )
    assert result["publication"] is False


def test_bounded_probe_process_times_out_allowlisted_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "lite_probe_child.py"
    child.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
    monkeypatch.setitem(host.run_bounded_probe_process.__globals__, "__file__", str(tmp_path / "e2e_real_model_host.py"))
    with pytest.raises(TimeoutError, match="exceeded"):
        host.run_bounded_probe_process(
            command=[sys.executable, str(child)], request=_probe_revision_binding(), timeout_seconds=0.05
        )


def test_repeated_benchmark_excludes_warmup_and_reports_median_observed_rate() -> None:
    runs = iter(
        [
            {"actual_tokens": 99, "elapsed_s": 99.0, "text": "warmup"},
            {"actual_tokens": 2, "elapsed_s": 0.4, "text": "measured-a"},
            {"actual_tokens": 3, "elapsed_s": 0.9, "text": "measured-b"},
        ]
    )

    result = host.benchmark_repeated_arm(
        run_once=lambda: next(runs),
        repeats=3,
        warmup_count=1,
    )

    assert result["text"] == "measured-b"
    assert result["benchmark"]["warmup_count"] == 1
    assert result["benchmark"]["measured_repeats"] == 2
    assert result["benchmark"]["actual_tokens"] == 5
    assert result["tok_per_s"] == pytest.approx(4.0)


def test_invalid_host_evidence_keeps_last_known_good_output(tmp_path: Path) -> None:
    destination = tmp_path / "evidence.json"
    destination.write_text('{"state":"last-good"}', encoding="utf-8")

    with pytest.raises(validation.ValidationError, match="immutable provenance"):
        host.publish_host_validation_evidence(
            destination=destination,
            candidate={"model_revision": "main"},
            approved_output_root=tmp_path,
        )

    assert destination.read_text(encoding="utf-8") == '{"state":"last-good"}'


def test_main_preflights_before_runtime_load_and_publishes_through_host_bridge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    manifest = validation.ValidationManifest.create(
        model_revision="a" * 64,
        corpus_revision="b" * 64,
        tokenizer_revision="c" * 64,
    )
    model = type("Model", (), {"model": type("Inner", (), {"layers": [object()]})()})()
    tokenizer = type("Tokenizer", (), {"vocab_size": 7})()

    def result(**values: object) -> dict[str, object]:
        return {
            "text": "answer",
            "actual_tokens": 2,
            "elapsed_s": 1.0,
            "tok_per_s": 2.0,
            "peak_rss_mb": 3.0,
            "kv_fp16_bytes": 4,
            "kv_turbo_bytes": 2,
            "kv_saved_bytes": 2,
            "kv_saved_pct": 50.0,
            "compact_layers": 1,
            "num_layers": 1,
            "config": "fake",
            **values,
        }

    monkeypatch.setattr(
        host,
        "validation_manifest_for_workload",
        lambda *, workload: events.append("preflight") or manifest,
    )
    monkeypatch.setattr(host, "import_stack", lambda: events.append("import-stack") or (None,) * 4)
    monkeypatch.setattr(
        host,
        "load_model",
        lambda _path: events.append("load-model") or (model, tokenizer),
    )
    monkeypatch.setattr(host, "gen_baseline", lambda *_args: result(kv_saved_bytes=0, kv_saved_pct=0))
    monkeypatch.setattr(
        host,
        "gen_mlx_native",
        lambda *_args, **_kwargs: result(kv_fp16_bytes=0, kv_saved_bytes=0, kv_saved_pct=0),
    )

    turbo_manifests: list[object] = []

    def fake_turbo(*_args: object, manifest: object, **_kwargs: object) -> dict[str, object]:
        turbo_manifests.append(manifest)
        return result()

    monkeypatch.setattr(host, "gen_turboquant", fake_turbo)
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        host,
        "publish_host_validation_evidence",
        lambda **kwargs: published.append(kwargs) or tmp_path / "evidence.json",
    )
    monkeypatch.setattr(host, "EVIDENCE_OUTPUT", tmp_path / "evidence.json", raising=False)

    with pytest.raises(host.TeacherForcedCapabilityError, match="TurboKVCacheLite"):
        host.main()
    assert published == []
    assert "load-model" not in events

    comparison = host.compare_teacher_forced_nll(
        token_count=2,
        baseline_total_nll=2.0,
        compacted_total_nll=2.2,
    )
    scorer_calls: list[dict[str, object]] = []
    host.main(
        teacher_forced_scorer=lambda **kwargs: scorer_calls.append(kwargs) or comparison
    )

    assert events.index("preflight") < events.index("import-stack") < events.index("load-model")
    assert turbo_manifests == [manifest] * 6
    assert len(published) == 1
    assert scorer_calls[0]["workload"].teacher_forced_continuation
    assert published[0]["candidate"]["model_revision"] == manifest.model_revision
    assert published[0]["candidate"]["teacher_forced"]["token_count"] == 2
    assert published[0]["approved_output_root"] == tmp_path


def test_checked_in_workload_sha_is_bound_as_the_accurately_named_corpus_revision() -> None:
    workload = host.load_benchmark_workload()
    payload = host.BENCHMARK_WORKLOAD_PATH.read_bytes()

    assert workload.name == "fibonacci-teacher-forced-v1"
    assert workload.kind == "checked_in_benchmark_workload"
    assert workload.revision == hashlib.sha256(payload).hexdigest()
    assert workload.prompt == "def fibonacci(n):\n    "

    manifest = host.validation_manifest_for_workload(
        {
            "PHENO_MODEL_REVISION": "a" * 64,
            "PHENO_TOKENIZER_REVISION": "b" * 64,
        },
        workload=workload,
    )
    assert manifest.corpus_revision == workload.revision

    with pytest.raises(validation.ValidationError, match="corpus_revision"):
        host.validation_manifest_for_workload(
            {
                "PHENO_MODEL_REVISION": "a" * 64,
                "PHENO_TOKENIZER_REVISION": "b" * 64,
                "PHENO_CORPUS_REVISION": "c" * 64,
            },
            workload=workload,
        )

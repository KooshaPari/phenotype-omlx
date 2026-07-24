#!/usr/bin/env python3
"""Tests for harbor_to_cockpit adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harbor_to_cockpit import (
    _build_cockpit_output,
    _elapsed_s,
    convert_job,
    convert_runs_root,
    discover_trials,
    load_job_result,
    main,
    parse_eval_name,
    trial_to_cell,
)


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


_DEFAULT_EVALS = {
    "oracle__adhoc": {
        "n_trials": 1,
        "n_errors": 0,
        "metrics": [{"mean": 1.0}],
        "pass_at_k": {},
        "reward_stats": {"reward": {"1.0": ["omlx-qwen35-policy__gS8UVyk"]}},
        "exception_stats": {},
    }
}


def _make_job_result(
    *,
    job_id: str = "test-job-001",
    evals: dict | None = None,
    started: str = "2026-07-22T22:39:39.651310",
    finished: str = "2026-07-22T22:40:30.147862",
) -> dict:
    return {
        "id": job_id,
        "started_at": started,
        "updated_at": finished,
        "finished_at": finished,
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
            "evals": _DEFAULT_EVALS if evals is None else evals,
            "n_input_tokens": None,
            "n_cache_tokens": None,
            "n_output_tokens": None,
            "cost_usd": None,
        },
        "run_policy": None,
        "resolved_provider": None,
        "fallback_applied": False,
    }


def _make_trial_result(
    *,
    trial_name: str = "omlx-qwen35-policy__gS8UVyk",
    task_name: str = "omlx/qwen35-policy",
    agent_name: str = "oracle",
    reward: float = 1.0,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    started: str = "2026-07-23T05:39:43.270305Z",
    finished: str = "2026-07-23T05:40:29.805562Z",
    agent_started: str = "2026-07-23T05:40:16.405600Z",
    agent_finished: str = "2026-07-23T05:40:16.941101Z",
    exception_info: dict | None = None,
) -> dict:
    return {
        "id": "e5b1ec20-54a9-4b58-b86a-6caa1491f1bb",
        "task_name": task_name,
        "trial_name": trial_name,
        "trial_uri": f"file:///tmp/{trial_name}",
        "task_id": {"path": f"/tasks/{trial_name}"},
        "source": None,
        "task_checksum": "abc123",
        "config": {
            "trial_name": trial_name,
            "trials_dir": "/tmp",
            "job_id": "test-job-001",
            "agent": {"name": agent_name, "model_name": None},
            "environment": {"type": "apple-container"},
        },
        "agent_info": {"name": agent_name, "version": "1.0.0", "model_info": None},
        "agent_result": {
            "n_input_tokens": tokens_in,
            "n_cache_tokens": None,
            "n_output_tokens": tokens_out,
            "cost_usd": None,
            "rollout_details": None,
            "metadata": None,
        },
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": exception_info,
        "started_at": started,
        "finished_at": finished,
        "environment_setup": {"started_at": started, "finished_at": started},
        "agent_setup": {"started_at": agent_started, "finished_at": agent_started},
        "agent_execution": {
            "started_at": agent_started,
            "finished_at": agent_finished,
        },
        "verifier": {
            "started_at": agent_finished,
            "finished_at": agent_finished,
        },
    }


class TestParseEvalName:
    def test_standard(self) -> None:
        assert parse_eval_name("oracle__adhoc") == ("oracle", "adhoc")

    def test_no_separator(self) -> None:
        assert parse_eval_name("mmlu") == ("unknown", "mmlu")

    def test_empty_agent(self) -> None:
        assert parse_eval_name("__adhoc") == ("unknown", "adhoc")

    def test_empty_suite(self) -> None:
        assert parse_eval_name("oracle__") == ("oracle", "adhoc")

    def test_custom_separator(self) -> None:
        assert parse_eval_name("a::b", sep="::") == ("a", "b")


class TestElapsedS:
    def test_valid(self) -> None:
        assert _elapsed_s("2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z") == 10.0

    def test_none_start(self) -> None:
        assert _elapsed_s(None, "2026-01-01T00:00:10Z") == 0.0

    def test_none_end(self) -> None:
        assert _elapsed_s("2026-01-01T00:00:00Z", None) == 0.0

    def test_both_none(self) -> None:
        assert _elapsed_s(None, None) == 0.0

    def test_negative_clamps(self) -> None:
        assert _elapsed_s("2026-01-01T00:00:10Z", "2026-01-01T00:00:00Z") == 0.0


class TestTrialToCell:
    def test_basic(self) -> None:
        trial = _make_trial_result(reward=0.85)
        cell = trial_to_cell(trial, "oracle", "adhoc")
        assert cell["suite"] == "adhoc"
        assert cell["variant"] == "oracle"
        assert cell["ok"] is True
        assert cell["pass_at_1"] == 0.85
        assert cell["scoring_method"] == "harbor_verifier"

    def test_error_trial(self) -> None:
        trial = _make_trial_result(exception_info={"type": "TimeoutError"})
        cell = trial_to_cell(trial, "oracle", "adhoc")
        assert cell["ok"] is False

    def test_no_reward(self) -> None:
        trial = _make_trial_result()
        trial["verifier_result"]["rewards"] = {}
        cell = trial_to_cell(trial, "oracle", "adhoc")
        assert cell["ok"] is False
        assert cell["pass_at_1"] == 0.0

    def test_tokens_per_second(self) -> None:
        trial = _make_trial_result(tokens_out=100)
        cell = trial_to_cell(trial, "oracle", "adhoc")
        assert cell["tokens_per_second"] > 180.0

    def test_metadata(self) -> None:
        trial = _make_trial_result()
        cell = trial_to_cell(trial, "oracle", "adhoc")
        assert cell["metadata"]["source"] == "harbor"
        assert cell["metadata"]["trial_name"] == "omlx-qwen35-policy__gS8UVyk"


class TestDiscoverTrials:
    def test_with_trial_dirs(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        trial = _make_trial_result()
        _write_json(tmp_path / "omlx-qwen35-policy__gS8UVyk" / "result.json", trial)
        found = discover_trials(tmp_path, job)
        assert len(found) == 1
        assert found[0]["trial_name"] == "omlx-qwen35-policy__gS8UVyk"

    def test_no_trial_dirs(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        found = discover_trials(tmp_path, job)
        assert found == []

    def test_skips_bad_json(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        bad_dir = tmp_path / "bad-trial"
        bad_dir.mkdir()
        (bad_dir / "result.json").write_text("not json {{{")
        found = discover_trials(tmp_path, job)
        assert found == []


class TestConvertJob:
    def test_with_trial_files(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        trial = _make_trial_result()
        _write_json(tmp_path / "trial-1" / "result.json", trial)
        result = convert_job(tmp_path)
        assert "summary" in result
        assert "cells" in result
        assert len(result["cells"]) == 1
        assert result["cells"][0]["variant"] == "oracle"
        assert result["summary"]["meta"]["model"] == "oracle"

    def test_fallback_synthesizes(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        result = convert_job(tmp_path)
        assert len(result["cells"]) == 1
        assert result["cells"][0]["pass_at_1"] == 1.0

    def test_missing_result_json(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_job(tmp_path)

    def test_empty_evals_no_trials(self, tmp_path: Path) -> None:
        job = _make_job_result(evals={})
        _write_json(tmp_path / "result.json", job)
        with pytest.raises(ValueError, match="No cells produced"):
            convert_job(tmp_path)

    def test_summary_structure(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        trial = _make_trial_result()
        _write_json(tmp_path / "trial-1" / "result.json", trial)
        result = convert_job(tmp_path)
        meta = result["summary"]["meta"]
        assert meta["n_cells"] == 1
        assert "oracle" in meta["variants"]
        assert result["summary"]["by_variant"]["oracle"]["n_cells"] == 1
        assert result["summary"]["by_variant"]["oracle"]["pass_at_1"] == 1.0


class TestMultiEval:
    def test_two_evals(self, tmp_path: Path) -> None:
        job = _make_job_result(
            evals={
                "oracle__mmlu": {
                    "n_trials": 1,
                    "n_errors": 0,
                    "metrics": [{"mean": 0.9}],
                    "pass_at_k": {},
                    "reward_stats": {},
                    "exception_stats": {},
                },
                "oracle__ifeval": {
                    "n_trials": 1,
                    "n_errors": 0,
                    "metrics": [{"mean": 0.7}],
                    "pass_at_k": {},
                    "reward_stats": {},
                    "exception_stats": {},
                },
            }
        )
        _write_json(tmp_path / "result.json", job)
        t1 = _make_trial_result(
            trial_name="mmlu-trial-1",
            task_name="oracle/mmlu",
            reward=0.9,
        )
        t2 = _make_trial_result(
            trial_name="ifeval-trial-1",
            task_name="oracle/ifeval",
            reward=0.7,
        )
        _write_json(tmp_path / "mmlu-trial-1" / "result.json", t1)
        _write_json(tmp_path / "ifeval-trial-1" / "result.json", t2)
        result = convert_job(tmp_path)
        assert len(result["cells"]) == 2

    def test_multiple_trials_same_eval(self, tmp_path: Path) -> None:
        job = _make_job_result(
            evals={
                "oracle__adhoc": {
                    "n_trials": 3,
                    "n_errors": 0,
                    "metrics": [{"mean": 0.8}],
                    "pass_at_k": {},
                    "reward_stats": {},
                    "exception_stats": {},
                },
            }
        )
        _write_json(tmp_path / "result.json", job)
        for i in range(3):
            t = _make_trial_result(trial_name=f"trial-{i}", reward=0.6 + i * 0.1)
            _write_json(tmp_path / f"trial-{i}" / "result.json", t)
        result = convert_job(tmp_path)
        assert len(result["cells"]) == 3
        scores = sorted(c["pass_at_1"] for c in result["cells"])
        assert scores == [0.6, 0.7, 0.8]


class TestConvertRunsRoot:
    def test_multiple_jobs(self, tmp_path: Path) -> None:
        for idx in range(2):
            d = tmp_path / f"job-{idx}"
            job = _make_job_result(job_id=f"job-{idx}")
            _write_json(d / "result.json", job)
            trial = _make_trial_result(reward=0.5 + idx * 0.25)
            _write_json(d / "trial-1" / "result.json", trial)
        result = convert_runs_root(tmp_path)
        assert len(result["cells"]) == 2

    def test_skips_non_job_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-job").mkdir()
        (tmp_path / "not-a-job" / "readme.txt").write_text("hello")
        job_dir = tmp_path / "real-job"
        job = _make_job_result()
        _write_json(job_dir / "result.json", job)
        trial = _make_trial_result()
        _write_json(job_dir / "trial-1" / "result.json", trial)
        result = convert_runs_root(tmp_path)
        assert len(result["cells"]) == 1


class TestCLI:
    def test_stdout(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        trial = _make_trial_result()
        _write_json(tmp_path / "trial-1" / "result.json", trial)
        ret = main([str(tmp_path)])
        assert ret == 0
        parsed = json.loads(capsys.readouterr().out)
        assert len(parsed["cells"]) == 1

    def test_file_output(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        trial = _make_trial_result()
        _write_json(tmp_path / "trial-1" / "result.json", trial)
        out = tmp_path / "output.json"
        ret = main([str(tmp_path), "-o", str(out)])
        assert ret == 0
        assert out.exists()
        assert len(json.loads(out.read_text())["cells"]) == 1

    def test_invalid_dir(self, tmp_path: Path) -> None:
        assert main([str(tmp_path / "nonexistent")]) == 1

    def test_all_flag(self, tmp_path: Path) -> None:
        for i in range(2):
            d = tmp_path / f"run-{i}"
            _write_json(d / "result.json", _make_job_result(job_id=f"r{i}"))
            _write_json(
                d / "t1" / "result.json",
                _make_trial_result(reward=float(i)),
            )
        out = tmp_path / "out.json"
        ret = main([str(tmp_path), "--all", "-o", str(out)])
        assert ret == 0
        assert len(json.loads(out.read_text())["cells"]) == 2


class TestCockpitContract:
    def test_cell_required_keys(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        _write_json(tmp_path / "t1" / "result.json", _make_trial_result())
        result = convert_job(tmp_path)
        cell = result["cells"][0]
        for key in [
            "suite",
            "task_id",
            "difficulty",
            "variant",
            "ok",
            "wall_clock_s",
            "pass_at_1",
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
        ]:
            assert key in cell, f"missing key: {key}"

    def test_summary_required_keys(self, tmp_path: Path) -> None:
        job = _make_job_result()
        _write_json(tmp_path / "result.json", job)
        _write_json(tmp_path / "t1" / "result.json", _make_trial_result())
        result = convert_job(tmp_path)
        meta = result["summary"]["meta"]
        for key in ["model", "n_suites", "n_cells", "variants", "difficulty_mix"]:
            assert key in meta, f"missing meta key: {key}"
        bv = result["summary"]["by_variant"]
        assert "oracle" in bv
        for key in ["n_cells", "pass_at_1", "mean_wall_clock_s", "ok_count"]:
            assert key in bv["oracle"], f"missing by_variant key: {key}"

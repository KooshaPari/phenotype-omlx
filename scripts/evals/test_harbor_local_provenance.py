"""Contracts for the explicit, non-exporting Harbor provenance path."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "evals" / "harbor"))

import harbor_local_provenance
from harbor_local_provenance import convert_local_harbor_run, main

SOURCE_HEAD = "a" * 40


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    _write(
        run / "result.json",
        {
            "id": "local-job",
            "started_at": "2026-07-26T00:00:00Z",
            "finished_at": "2026-07-26T00:01:00Z",
            "stats": {
                "evals": {
                    "oracle__omlx-niah-api-smoke": {
                        "n_trials": 1,
                        "metrics": [{"mean": 1.0}],
                    }
                },
                "n_completed_trials": 1,
            },
        },
    )
    _write(
        run / "trial" / "result.json",
        {
            "id": "trial-id",
            "trial_name": "trial-id",
            "task_name": "omlx-niah-api-smoke",
            "agent_info": {"name": "oracle", "version": "1"},
            "agent_result": {"n_input_tokens": 3, "n_output_tokens": 5},
            "verifier_result": {"rewards": {"reward": 1.0}},
            "started_at": "2026-07-26T00:00:00Z",
            "finished_at": "2026-07-26T00:01:00Z",
            "config": {"job_id": "local-job"},
        },
    )
    return run


def test_local_conversion_preserves_source_and_validates(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    source = (run / "result.json").read_bytes()
    report, validation = convert_local_harbor_run(
        run, model="Qwen3.5-0.8B", commit_sha=SOURCE_HEAD
    )
    assert validation.valid, validation.errors
    assert (run / "result.json").read_bytes() == source
    assert report["telemetry"] == {"mode": "local_only", "remote_exported": False}
    assert report["run"]["model"] == "Qwen3.5-0.8B"
    assert report["suites"][0]["evidence_label"] == "live_verified"
    assert (
        report["hash_chain"]["task_ids_sorted_sha256"]
        == hashlib.sha256(b"trial-id").hexdigest()
    )


def test_local_report_binds_all_required_source_digests(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    report, validation = convert_local_harbor_run(
        run, model="Qwen3.5-0.8B", commit_sha=SOURCE_HEAD
    )

    assert validation.valid, validation.errors
    provenance = report["source_provenance"]
    assert provenance["source_head"] == SOURCE_HEAD
    assert (
        provenance["result_json"]["sha256"]
        == hashlib.sha256((run / "result.json").read_bytes()).hexdigest()
    )
    assert (
        provenance["job_yaml"]["sha256"]
        == hashlib.sha256(
            harbor_local_provenance.HARBOR_JOB_CONFIG.read_bytes()
        ).hexdigest()
    )
    assert (
        provenance["model_config"]["sha256"]
        == hashlib.sha256(harbor_local_provenance.MODEL_CONFIG.read_bytes()).hexdigest()
    )


def test_local_conversion_fails_closed_when_required_model_config_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harbor_local_provenance, "MODEL_CONFIG", tmp_path / "missing.json"
    )

    with pytest.raises(FileNotFoundError, match="required provenance source"):
        convert_local_harbor_run(
            _run_dir(tmp_path), model="Qwen3.5-0.8B", commit_sha=SOURCE_HEAD
        )


def test_local_conversion_rejects_noncanonical_source_head(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="40-character lowercase Git SHA"):
        convert_local_harbor_run(
            _run_dir(tmp_path), model="Qwen3.5-0.8B", commit_sha="source-head"
        )


def test_cli_refuses_to_overwrite_or_follow_existing_report_output(
    tmp_path: Path,
) -> None:
    run = _run_dir(tmp_path)
    protected = tmp_path / "protected.json"
    protected.write_text("preserve-me", encoding="utf-8")
    output = tmp_path / "evaluation_report.local.json"
    output.symlink_to(protected)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                str(run),
                "--model",
                "Qwen3.5-0.8B",
                "--output",
                str(output),
            ]
        )

    assert exc_info.value.code == 2
    assert protected.read_text(encoding="utf-8") == "preserve-me"


def test_local_conversion_discovers_single_timestamped_job_directory(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "harbor-output"
    run = _run_dir(tmp_path)
    timestamped_run = output_root / "2026-07-26__06-25-42"
    timestamped_run.parent.mkdir()
    run.rename(timestamped_run)
    run = timestamped_run
    source = (run / "result.json").read_bytes()

    report, validation = convert_local_harbor_run(
        output_root, model="Qwen3.5-0.8B", commit_sha=SOURCE_HEAD
    )

    assert validation.valid, validation.errors
    assert report["telemetry"] == {"mode": "local_only", "remote_exported": False}
    assert (run / "result.json").read_bytes() == source


def test_local_conversion_rejects_ambiguous_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "harbor-output"
    for name in ("2026-07-26__06-25-42", "2026-07-26__06-30-00"):
        run = _run_dir(tmp_path / name)
        output_root.mkdir(exist_ok=True)
        run.rename(output_root / name)

    with pytest.raises(ValueError, match="multiple Harbor job directories"):
        convert_local_harbor_run(
            output_root, model="Qwen3.5-0.8B", commit_sha=SOURCE_HEAD
        )


def test_local_report_has_no_remote_trace_or_session_fields(tmp_path: Path) -> None:
    report, _ = convert_local_harbor_run(
        _run_dir(tmp_path), model="Qwen3.5-0.8B", commit_sha=SOURCE_HEAD
    )
    serialized = json.dumps(report).lower()
    assert "langfuse" not in serialized
    assert "trace" not in serialized
    assert "session" not in serialized


def test_strict_wrapper_still_requires_langfuse_credentials(tmp_path: Path) -> None:
    portage = tmp_path / "portage"
    (portage / "packages" / "harbor-langfuse" / "src").mkdir(parents=True)
    env = {
        "PORTAGE_ROOT": str(portage),
        "HARBOR_ENV": "apple-container",
        "PATH": "/usr/bin:/bin",
    }
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/evals/run_via_harbor.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required" in proc.stderr

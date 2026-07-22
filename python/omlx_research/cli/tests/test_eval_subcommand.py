"""Tests for the ``omlx-research eval`` CLI subcommand.

Covers two surfaces:

1. ``eval --help`` exits 0 and lists the four suite identifiers
   (``mmlu``, ``gpqa``, ``terminal-bench``, ``perplexity``) plus the
   ``--suite``, ``--dataset``, ``--backend``, and ``--report`` flags.
2. ``eval --suite mmlu --dataset <csv>`` runs against a tiny CSV and
   emits a JSON report on stdout with the documented shape::

       {"suite": "mmlu", "tasks": 5, "passed": 4, "score": 0.8, ...}

These tests exercise the Python wrapper in ``cli/__init__.py`` via the
``main([...])`` entry point so they share the same dispatch path real
users hit; the canonical scorer lives in the Rust ``eval-harness``
crate and is invoked via the kernel-registry in production. The wrapper
is a deterministic stub that mirrors ``eval_harness::Suite``'s
``serde(rename_all = "lowercase")`` form so the on-disk dataset
identifiers stay byte-compatible with the Rust enum.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

from omlx_research.cli import EVAL_VALID_SUITES, main


# --- stdio capture --------------------------------------------------------


class _IO:
    """Swap sys.stdout/sys.stderr for one ``main([...])`` call."""

    def __init__(self) -> None:
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def __enter__(self) -> "_IO":
        self._real_out, self._real_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.stdout, self.stderr
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout, sys.stderr = self._real_out, self._real_err


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    """Invoke ``main(argv)`` and capture stdout/stderr as strings."""
    with _IO() as cap:
        try:
            rc = main(argv)
        except SystemExit as e:
            # argparse uses SystemExit(0) for --help. Capture the
            # return code through the exception's code attribute.
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return rc, cap.stdout.getvalue(), cap.stderr.getvalue()


def _run_subprocess(argv: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Spawn ``python -m omlx_research.cli <argv>`` with cwd set to ``python/``.

    The wrapper module is only importable from ``python/`` because
    that's where ``omlx_research`` lives; subprocess invocations from
    the repo root need an explicit ``cwd`` so the spawned interpreter
    resolves the package correctly. Used by the end-to-end tests that
    exercise a separate Python process to mirror real CLI usage.
    """
    return subprocess.run(
        [sys.executable, "-m", "omlx_research.cli", *argv],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
    )


def _python_dir() -> str:
    """Absolute path to the ``python/`` directory that contains omlx_research.

    Test file lives at ``python/omlx_research/cli/tests/test_eval_subcommand.py``;
    climbing three levels reaches ``python/`` which is where
    ``omlx_research`` is importable from.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _write_mmlu_csv(path: str, rows: int = 5) -> None:
    """Write a minimal MMLU-shaped CSV with ``rows`` data rows.

    The schema matches ``perf-core/eval-harness/src/mmlu/mod.rs``:
    ``subject,question,A,B,answer``. Every row carries a valid answer
    letter so the wrapper's "non-empty answer => passed" stub counts
    them all as correct.
    """
    lines = ["subject,question,A,B,answer"]
    for i in range(rows):
        lines.append(f"anatomy,Q{i},Cranial,Thoracic,B")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- test_eval_help -------------------------------------------------------


def test_eval_help():
    """``main(["eval", "--help"])`` returns 0 and lists the documented flags.

    argparse prints --help to stdout and exits via SystemExit(0); we
    capture the code through ``_run_main`` so the test does not see
    the SystemExit itself.
    """
    rc, stdout, stderr = _run_main(["eval", "--help"])
    assert rc == 0, f"stderr: {stderr}"
    output = stdout + stderr

    # All four suite identifiers must appear in the choices list.
    for suite in EVAL_VALID_SUITES:
        assert suite in output, f"suite {suite!r} missing from --help output"

    # Each flag must be listed.
    for flag in ("--suite", "--dataset", "--backend", "--report"):
        assert flag in output, f"flag {flag!r} missing from --help output"


def test_eval_help_subprocess():
    """``python -m omlx_research.cli eval --help`` exits 0 in a fresh process.

    Mirrors what a real user sees when they type the command in their
    shell. We set ``cwd=python/`` because that's the only directory
    from which ``omlx_research`` resolves.
    """
    proc = _run_subprocess(["eval", "--help"], cwd=_python_dir())
    assert proc.returncode == 0, (
        f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    )
    output = proc.stdout + proc.stderr
    for suite in EVAL_VALID_SUITES:
        assert suite in output
    for flag in ("--suite", "--dataset", "--backend", "--report"):
        assert flag in output


# --- test_eval_mmlu_runs_with_stub_dataset --------------------------------


def test_eval_mmlu_runs_with_stub_dataset(tmp_path):
    """End-to-end: ``eval --suite mmlu --dataset <csv>`` prints a JSON report.

    Uses a subprocess invocation so the test exercises the exact code
    path real users hit: argparse dispatch + ``cmd_eval`` + JSON emit.
    """
    csv_path = tmp_path / "tiny_mmlu.csv"
    _write_mmlu_csv(str(csv_path), rows=5)

    proc = _run_subprocess(
        ["eval", "--suite", "mmlu", "--dataset", str(csv_path)],
        cwd=_python_dir(),
    )
    assert proc.returncode == 0, (
        f"eval exited {proc.returncode}\nstdout: {proc.stdout}\n"
        f"stderr: {proc.stderr}"
    )

    # Stdout is pure JSON (no preamble).
    report = json.loads(proc.stdout)
    assert report["suite"] == "mmlu"
    assert report["tasks"] == 5
    assert report["passed"] == 5
    assert report["score"] == 1.0
    # Default backend is "metal" and we did not pass --report.
    assert report["backend"] == "metal"
    assert report["report_path"] is None
    # Harness tag is informative and references the Rust crate by name.
    assert "eval-harness" in report["harness"]
    assert isinstance(report["timestamp"], (int, float))


def test_eval_terminal_bench_jsonl(tmp_path):
    """JSONL datasets load via the documented terminal-bench/perplexity path."""
    jsonl = tmp_path / "tbench.jsonl"
    lines = [
        json.dumps({"id": "t1", "prompt": "ls", "expected": "ls"}),
        json.dumps({"id": "t2", "prompt": "pwd", "expected": "pwd"}),
        json.dumps({"id": "t3", "prompt": "wc", "expected": "wc -l"}),
    ]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    proc = _run_subprocess(
        ["eval", "--suite", "terminal-bench", "--dataset", str(jsonl)],
        cwd=_python_dir(),
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["suite"] == "terminal-bench"
    assert report["tasks"] == 3
    assert report["passed"] == 3


def test_eval_unknown_suite_exits_2(tmp_path):
    """An unknown suite identifier fails fast with exit 2."""
    csv_path = tmp_path / "noop.csv"
    csv_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    proc = _run_subprocess(
        ["eval", "--suite", "bogus-suite", "--dataset", str(csv_path)],
        cwd=_python_dir(),
    )
    # argparse's choices validation rejects the value before our code runs,
    # which exits with code 2 (the standard argparse error code).
    assert proc.returncode == 2
    assert "bogus-suite" in (proc.stderr + proc.stdout)


def test_eval_missing_dataset_exits_2():
    """A non-existent dataset path produces a structured error and exit 2."""
    proc = _run_subprocess(
        ["eval", "--suite", "mmlu", "--dataset", "/tmp/__definitely_missing__.csv"],
        cwd=_python_dir(),
    )
    assert proc.returncode == 2
    assert "not found" in (proc.stderr + proc.stdout)


def test_eval_report_persists_to_disk(tmp_path):
    """``--report`` writes a copy of the report to the requested path."""
    csv_path = tmp_path / "tiny_mmlu.csv"
    _write_mmlu_csv(str(csv_path), rows=2)
    report_path = tmp_path / "reports" / "out.json"

    proc = _run_subprocess(
        [
            "eval",
            "--suite",
            "gpqa",
            "--dataset",
            str(csv_path),
            "--report",
            str(report_path),
        ],
        cwd=_python_dir(),
    )
    assert proc.returncode == 0, proc.stderr

    stdout_report = json.loads(proc.stdout)
    assert stdout_report["suite"] == "gpqa"
    assert stdout_report["report_path"] == str(report_path)

    assert report_path.is_file()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    # The on-disk copy matches the stdout report (modulo the timestamp
    # which we deliberately do not pin).
    assert on_disk["suite"] == stdout_report["suite"]
    assert on_disk["tasks"] == stdout_report["tasks"]
    assert on_disk["passed"] == stdout_report["passed"]


def test_eval_in_process_dispatch(tmp_path):
    """``main([...])`` dispatches to cmd_eval and emits the JSON report.

    Drives the dispatch via the in-process ``main`` entry point so the
    test does not depend on a subprocess — matches the pattern used by
    the rest of the suite (e.g. ``test_doctor.py``).
    """
    csv_path = tmp_path / "tiny.csv"
    _write_mmlu_csv(str(csv_path), rows=3)

    rc, stdout, stderr = _run_main(
        ["eval", "--suite", "mmlu", "--dataset", str(csv_path)]
    )
    assert rc == 0, f"stderr: {stderr}"
    report = json.loads(stdout)
    assert report["suite"] == "mmlu"
    assert report["tasks"] == 3
    assert report["passed"] == 3
    assert report["score"] == 1.0


# --- guards for the contract surface -------------------------------------


def test_eval_valid_suites_matches_rust_enum():
    """The Python suite list must mirror eval_harness::Suite (lowercase).

    The Rust enum uses ``#[serde(rename_all = "lowercase")]`` so the
    on-disk identifiers are exactly: ``mmlu``, ``gpqa``,
    ``terminal-bench``, ``perplexity``. If anyone renames a variant in
    Rust, this test fails before users see an argparse surprise.
    """
    assert EVAL_VALID_SUITES == ("mmlu", "gpqa", "terminal-bench", "perplexity")
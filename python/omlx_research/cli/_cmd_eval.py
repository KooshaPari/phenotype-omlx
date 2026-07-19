"""``omlx-research eval`` subcommand implementation — moved out of
``cli/__init__.py`` in turn-9's module-size sweep.

The eval-harness subcommand is the only non-trivial inline block in
``cli/__init__.py``: it owns the ``EVAL_VALID_SUITES`` constant (mirrors
the Rust ``eval_harness::Suite`` enum's
``#[serde(rename_all = "lowercase")]`` form), the dataset loader
(``_eval_load_dataset`` — CSV for multiple-choice suites, JSONL for
open-ended), the deterministic stub scorer (``_eval_stub_score``), and
the JSON-report emit path (``cmd_eval``). The other inline
``cmd_<x>`` stubs in ``cli/__init__.py`` are deliberately tiny (each
<40 lines, mostly dispatch glue) and stay put to keep their call site
in :func:`omlx_research.cli.main` next to the corresponding argparse
parser setup.

Public contract (re-exported by ``cli/__init__.py`` so existing
importers continue to work):

- :data:`EVAL_VALID_SUITES` — ``("mmlu", "gpqa", "terminal-bench", "perplexity")``
- :func:`cmd_eval` — the ``eval`` subcommand handler

The Rust scorer lives in ``perf-core/eval-harness/`` and is invoked
through the kernel-registry binding in production; the Python wrapper
is a deterministic stub that mirrors what ``EvalHarness::run`` marks as
"answer present" for the multiple-choice suites.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional


EVAL_VALID_SUITES: tuple[str, ...] = (
    "mmlu",
    "gpqa",
    "terminal-bench",
    "perplexity",
)


def _eval_load_dataset(suite: str, dataset_path: str) -> list[dict]:
    """Load an eval-harness dataset off disk in the documented shape.

    The Rust loaders (``perf-core/eval-harness/src/{dataset,mmlu,gpqa}.rs``)
    accept CSV for the multiple-choice suites (mmlu, gpqa) and JSONL for
    the open-ended ones (terminal-bench, perplexity). We mirror that
    split here so the wrapper's contract matches what the Rust crate
    expects on disk.

    Returns a list of raw row dicts; the wrapper is a stub so we do not
    validate per-suite schemas — the row count drives the report, and
    the schema check lives in the Rust loader.
    """
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"dataset not found: {dataset_path} (suite={suite})"
        )

    if suite in ("terminal-bench", "perplexity"):
        # JSONL: one JSON object per line.
        rows: list[dict] = []
        with open(dataset_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"dataset {dataset_path} line {lineno}: "
                        f"invalid JSON: {e}"
                    ) from e
        return rows

    # CSV: dict-reader on the header row.
    import csv as _csv  # local import — keeps startup lean
    with open(dataset_path, "r", encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        return [dict(row) for row in reader]


def _eval_stub_score(rows: list[dict], suite: str) -> int:
    """Deterministic stub: count rows with a non-empty answer field.

    The real scorer lives in Rust; this stub exists so the wrapper has
    an honest end-to-end contract for tests and CI smoke runs. It
    counts the same rows the Rust ``EvalHarness::run`` would mark as
    "answer present" for the multiple-choice suites.

    The ``answer`` field is canonical for the multiple-choice suites
    (mmlu/gpqa); the open-ended suites (terminal-bench/perplexity)
    store the expected completion under ``expected``. The stub
    accepts either so the wrapper's JSON-report contract is uniform
    across all four suites.
    """
    del suite  # unused in the stub; reserved for suite-specific heuristics
    score = 0
    for r in rows:
        answer = str(r.get("answer", "")).strip()
        if not answer:
            answer = str(r.get("expected", "")).strip()
        if answer:
            score += 1
    return score


def cmd_eval(args: argparse.Namespace) -> int:
    """Run an eval-harness suite against a local dataset file.

    Loads the dataset (CSV for multiple-choice, JSONL for open-ended),
    asks the kernel-registry's eval-harness binding to score it (stub
    today: count rows with a non-empty answer), and prints a JSON
    report on stdout. ``--report PATH`` also persists the report to
    disk so it can be archived alongside the run.
    """
    suite: str = args.suite
    dataset_path: str = args.dataset
    backend: str = args.backend
    report_path: Optional[str] = args.report

    try:
        rows = _eval_load_dataset(suite, dataset_path)
    except FileNotFoundError as e:
        # Surface a structured argparse-style error and exit 2 — the
        # standard exit code for CLI usage errors so scripts wrapping
        # the CLI can distinguish "bad input" from "internal failure".
        print(f"omlx-research eval: error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"omlx-research eval: error: {e}", file=sys.stderr)
        return 2

    tasks = len(rows)
    passed = _eval_stub_score(rows, suite)
    score = (passed / tasks) if tasks else 0.0

    report = {
        "suite": suite,
        "tasks": tasks,
        "passed": passed,
        "score": round(score, 6),
        "backend": backend,
        "harness": "eval-harness (Rust crate via kernel-registry; stub scorer in Python wrapper)",
        "dataset": dataset_path,
        "report_path": report_path,
        "timestamp": time.time(),
    }
    payload = json.dumps(report)
    print(payload)

    if report_path is not None:
        out_dir = os.path.dirname(os.path.abspath(report_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")

    return 0

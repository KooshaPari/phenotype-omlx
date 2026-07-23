"""Unit tests for Langfuse evaluator helpers (no live API)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evals"
    / "run_langfuse_evaluators.py"
)


def _load_mod():
    spec = importlib.util.spec_from_file_location("run_langfuse_evaluators", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_judge_payload_ok():
    mod = _load_mod()
    score, reason = mod.parse_judge_payload('{"score": 0.8, "reason": "partial"}')
    assert score == 0.8
    assert "partial" in reason


def test_parse_judge_payload_unparseable():
    mod = _load_mod()
    score, reason = mod.parse_judge_payload("no json here")
    assert score == 0.0
    assert reason.startswith("unparseable:")


def test_default_data_path_prefers_historical_v5(monkeypatch: pytest.MonkeyPatch):
    mod = _load_mod()
    monkeypatch.delenv("BENCH_DATA", raising=False)
    path = mod.default_data_path()
    assert path.is_file()
    assert path.name == "run-v5-qwen35-08b.json" or path.name == "smoke_results.json"

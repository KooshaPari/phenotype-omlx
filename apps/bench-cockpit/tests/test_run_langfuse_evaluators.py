"""Unit tests for Langfuse evaluator helpers (no live API)."""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
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


def _write_dotenv(workspace: Path, content: str, *, mode: int = 0o600) -> Path:
    env = workspace / ".env"
    env.write_text(content)
    env.chmod(mode)
    return env


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


def test_load_dotenv_namespace_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_mod()
    _write_dotenv(
        tmp_path,
        "# comment line\n"
        "\n"
        "PORTAGE_ROOT=/tmp/portage\n"
        "LANGFUSE_PUBLIC_KEY=pk-test\n"
        "LANGFUSE_SECRET_KEY=sk-test\n"
        "OBSERVABILITY_BACKEND=langfuse\n"
        "OPENAI_API_KEY=sk-leaked\n"
        "ANTHROPIC_API_KEY=sk-leaked\n"
        "MLX_SERVER_URL=http://leaked\n"
        "NOT_IN_PREFIX=hello\n",
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for k in (
        "PORTAGE_ROOT",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "OBSERVABILITY_BACKEND",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MLX_SERVER_URL",
        "NOT_IN_PREFIX",
    ):
        monkeypatch.delenv(k, raising=False)
    mod._load_dotenv()
    assert os.environ["PORTAGE_ROOT"] == "/tmp/portage"
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"
    assert os.environ["OBSERVABILITY_BACKEND"] == "langfuse"
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MLX_SERVER_URL", "NOT_IN_PREFIX"):
        assert k not in os.environ, f"{k} must not be loaded from .env"


def test_load_dotenv_refuses_permissive_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_mod()
    _write_dotenv(
        tmp_path,
        "PORTAGE_ROOT=/tmp/portage\n",
        mode=0o664,  # group-writable
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.delenv("PORTAGE_ROOT", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod._load_dotenv()
    assert any("perm" in str(w.message).lower() for w in caught)
    assert "PORTAGE_ROOT" not in os.environ


def test_load_dotenv_warns_on_empty_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_mod()
    _write_dotenv(
        tmp_path,
        "PORTAGE_ROOT=\nLANGFUSE_PUBLIC_KEY=pk-test\nLANGFUSE_SECRET_KEY=sk-test\n",
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for k in ("PORTAGE_ROOT", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod._load_dotenv()
    msgs = [str(w.message) for w in caught]
    assert any("empty token" in m and "PORTAGE_ROOT" in m for m in msgs)
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "pk-test"


def test_load_dotenv_does_not_clobber_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_mod()
    _write_dotenv(
        tmp_path,
        "LANGFUSE_PUBLIC_KEY=pk-from-dotenv\nLANGFUSE_SECRET_KEY=sk-test\n",
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-already-set")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    mod._load_dotenv()
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-already-set"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"


def test_load_dotenv_skips_comments_and_blanks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_mod()
    _write_dotenv(
        tmp_path,
        "# top comment\n"
        "\n"
        "   \n"
        "LANGFUSE_PUBLIC_KEY=pk-test\n"
        " # indented comment\n"
        "LANGFUSE_SECRET_KEY=sk-test\n"
        "=orphan\n"
        "no_equals_sign\n",
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    mod._load_dotenv()
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"

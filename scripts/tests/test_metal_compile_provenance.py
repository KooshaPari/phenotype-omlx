"""Guardrails for exact-head Metal compile provenance capture."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.record_metal_compile_provenance import _record


def test_rejects_provenance_output_inside_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(RuntimeError, match="outside the repository"):
        _record(repo, repo / "provenance.json", None)


def test_rejects_existing_external_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "provenance.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output already exists"):
        _record(repo, output, None)

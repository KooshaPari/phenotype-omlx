"""FR-5 / E1–E2 tests: Julia required (fail loud) + NIAH non-legacy paths.

Acceptance:
- E1: ``julia`` required on the eval/doctor path — FAIL (not WARN) if missing.
- E2: ``scripts/niah_benchmark.py`` must not use legacy ``phenotype-omlx/python``.
"""

from __future__ import annotations

import os
from pathlib import Path

from omlx_research.cli import _doctor_checks as checks_mod
from omlx_research.cli import _doctor_extra_niah as niah_extra
from omlx_research.cli._doctor_shared import FAIL, PASS, project_root


def test_julia_required_fails_when_missing(monkeypatch):
    """E1: missing ``julia`` on PATH ⇒ doctor check FAIL (loud)."""

    monkeypatch.setattr(niah_extra.shutil, "which", lambda _name: None)
    c = checks_mod.julia_required_on_eval_path()
    assert c.status == FAIL
    assert c.id == "julia_required_on_eval_path"
    assert "julia" in c.details.lower()


def test_julia_required_passes_when_present(monkeypatch):
    """E1: ``julia`` on PATH ⇒ PASS and surface the resolved binary."""

    monkeypatch.setattr(
        niah_extra.shutil, "which", lambda _name: "/opt/homebrew/bin/julia"
    )

    class _FakeProc:
        returncode = 0
        stdout = "julia version 1.12.6\n"
        stderr = ""

    monkeypatch.setattr(
        niah_extra.subprocess,
        "run",
        lambda *a, **kw: _FakeProc(),
    )
    c = checks_mod.julia_required_on_eval_path()
    assert c.status == PASS
    assert "/opt/homebrew/bin/julia" in c.details
    assert "1.12.6" in c.details


def test_niah_benchmark_source_has_no_legacy_python_path():
    """E2: committed NIAH script must not hardcode legacy absolute repos path."""

    root = Path(project_root())
    script = root / "scripts" / "niah_benchmark.py"
    text = script.read_text(encoding="utf-8")
    assert 'REPO / "phenotype-omlx/python"' not in text
    assert 'Path("/Users/kooshapari/CodeProjects/Phenotype/repos")' not in text
    assert 'ROOT / "python"' in text


def test_niah_benchmark_legacy_path_doctor_fails(monkeypatch, tmp_path):
    """Doctor FAIL when the on-disk NIAH script still embeds the legacy path."""

    bad = tmp_path / "niah_benchmark.py"
    bad.write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'REPO = Path("/Users/kooshapari/CodeProjects/Phenotype/repos")\n'
        'sys.path.insert(0, str(REPO / "phenotype-omlx/python"))\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(niah_extra, "_find_niah_benchmark", lambda: str(bad))
    c = niah_extra.niah_benchmark_non_legacy_path()
    assert c.status == FAIL
    assert "legacy" in c.details.lower() or "phenotype-omlx/python" in c.details


def test_niah_benchmark_legacy_path_doctor_passes_for_repo_relative(monkeypatch, tmp_path):
    """Doctor PASS when the script uses repo-relative ``<root>/python``."""

    good = tmp_path / "niah_benchmark.py"
    good.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        'sys.path.insert(0, str(ROOT / "python"))\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(niah_extra, "_find_niah_benchmark", lambda: str(good))
    c = niah_extra.niah_benchmark_non_legacy_path()
    assert c.status == PASS

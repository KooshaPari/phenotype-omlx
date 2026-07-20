"""FR-5 / E1–E4 tests: Julia, non-legacy paths, evidence-class discipline.

Acceptance:
- E1: ``julia`` required on the eval/doctor path — FAIL (not WARN) if missing.
- E2: ``scripts/niah_benchmark.py`` must not use legacy ``phenotype-omlx/python``.
- E4: committed ``niah_results.json`` must declare synthetic evidence class;
  live smoke must refuse to overwrite that envelope.
"""

from __future__ import annotations

import importlib.util
import json
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


def test_niah_results_committed_envelope_is_synthetic_not_live():
    """FR-5 E4: committed niah_results.json must declare synthetic evidence class."""

    root = Path(project_root())
    path = root / "niah_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("evidence_label") == "synthetic_target_rows"
    assert data.get("synthetic") is True
    assert data.get("reported") is True
    assert data.get("evidence_label") != "live_verified"
    assert len(data.get("targets") or []) >= 25


def test_niah_live_artifact_must_not_overwrite_envelope_name():
    """FR-5 E4: live smoke refuses to target niah_results.json."""

    script = Path(project_root()) / "scripts" / "niah_server_smoke.py"
    spec = importlib.util.spec_from_file_location("niah_server_smoke", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--output", "niah_results.json", "--server-url", "http://127.0.0.1:9"])
    assert rc == 2

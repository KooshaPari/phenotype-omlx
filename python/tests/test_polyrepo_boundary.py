"""Non-Git super-root boundary contracts."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_boundary() -> object:
    spec = importlib.util.spec_from_file_location(
        "polyrepo_boundary_under_test", ROOT / "scripts" / "polyrepo_boundary.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_non_git_root_accepts_root_without_git_metadata(tmp_path: Path) -> None:
    boundary = _load_boundary()
    result = boundary.check_non_git_root(tmp_path)
    assert result.ok is True
    assert result.reason == "ROOT_NOT_GIT"


def test_non_git_root_rejects_git_directory_or_worktree_file(tmp_path: Path) -> None:
    boundary = _load_boundary()
    (tmp_path / ".git").mkdir()
    assert boundary.check_non_git_root(tmp_path).ok is False

    (tmp_path / ".git").rmdir()
    (tmp_path / ".git").write_text("gitdir: /private/tmp/child/.git\n", encoding="utf-8")
    result = boundary.check_non_git_root(tmp_path)
    assert result.ok is False
    assert result.reason == "ROOT_GIT_METADATA"

"""Contracts for the sourceable phenotype-omlx environment helper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "phenotype-omlx-env.sh"


def _sourced_environment(cwd: Path, shell: str) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PHENOTYPE_OMLX_HOME", "REPOS_ROOT", "PHENOTYPE_OMLX_PYTHON"):
        environment.pop(name, None)
    completed = subprocess.run(
        [shell, "-c", 'source "$1" >/dev/null; env -0', shell, str(SCRIPT)],
        check=True,
        cwd=cwd,
        env=environment,
        capture_output=True,
    )
    return {
        name: value
        for entry in completed.stdout.decode().split("\0")
        if entry and (name := entry.partition("=")[0])
        for value in [entry.partition("=")[2]]
    }


@pytest.mark.parametrize("shell", ["bash", "zsh"])
def test_environment_script_resolves_recovered_checkout_and_offline_python(
    tmp_path: Path,
    shell: str,
) -> None:
    environment = _sourced_environment(tmp_path, shell)

    assert environment["PHENOTYPE_OMLX_HOME"] == str(ROOT)
    assert environment["REPOS_ROOT"] == str(ROOT.parent)
    assert environment["PHENOTYPE_OMLX_PYTHON"].endswith("python3.12")
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert str(ROOT / "python") in environment["PYTHONPATH"].split(":")
    assert environment["PATH"].split(":")[0] == str(ROOT / "cli" / "bin")


def test_launcher_help_resolves_its_own_checkout_from_an_unrelated_cwd(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PHENOTYPE_OMLX_HOME", None)

    completed = subprocess.run(
        [str(ROOT / "cli" / "bin" / "omlx-research"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()

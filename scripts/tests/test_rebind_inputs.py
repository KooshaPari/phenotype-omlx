"""Focused contracts for repository-relative candidate-rebind input helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import rebind_inputs
from scripts.rebind_inputs import CandidateRebindError, repository_file_descriptor


def test_rejects_repository_relative_read_outside_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"candidate artifact")
    monkeypatch.setattr(rebind_inputs.os, "name", "nt")

    with pytest.raises(CandidateRebindError, match="unsupported on this platform"):
        repository_file_descriptor(
            tmp_path,
            {
                "path": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
            "evidence artifact",
        )

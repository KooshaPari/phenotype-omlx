"""Focused tests for static fixture-source admission."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RECORDER = ROOT / "scripts" / "record_metal_device_fixture.py"


def _load_recorder():
    if str(RECORDER.parent) not in sys.path:
        sys.path.insert(0, str(RECORDER.parent))
    spec = importlib.util.spec_from_file_location(
        "record_metal_device_fixture", RECORDER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selected_fixture_requires_its_own_test_and_ignore_attributes(
    tmp_path: Path,
) -> None:
    recorder = _load_recorder()
    source = (
        tmp_path / "perf-core" / "metal-runtime" / "tests" / "diffusion_dispatch.rs"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        """
#[test]
#[ignore = "different fixture"]
fn unrelated_fixture() {}

#[test]
fn diffusion_three_stage_fixture_matches_oracle() {
    let _ = "METAL_RUNTIME_TEST_ARTIFACT METAL_RUNTIME_TEST_MANIFEST";
}
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="fixture source contract unavailable"):
        recorder._require_fixture_source_contract(tmp_path, "diffusion")

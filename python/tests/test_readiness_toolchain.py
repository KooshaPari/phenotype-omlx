"""Rust toolchain selection contracts for readiness gates."""

import importlib.util
import os
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_readiness() -> object:
    spec = importlib.util.spec_from_file_location(
        "phenotype_readiness_toolchain", ROOT / "scripts" / "readiness_check.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_cargo_prefers_newest_stable_host_toolchain(tmp_path: Path) -> None:
    """Avoid an older Cargo that cannot parse the workspace lockfile."""

    for version in ("1.74.0", "1.96.0"):
        cargo = tmp_path / "toolchains" / f"{version}-aarch64-apple-darwin" / "bin" / "cargo"
        cargo.parent.mkdir(parents=True)
        cargo.write_text("#!/bin/sh\n", encoding="utf-8")
        cargo.chmod(cargo.stat().st_mode | stat.S_IXUSR)

    readiness = _load_readiness()
    resolved = readiness.resolve_cargo(
        env={}, which=lambda _name: None, rustup_home=tmp_path, host_triple="aarch64-apple-darwin"
    )

    assert resolved == str(
        tmp_path / "toolchains" / "1.96.0-aarch64-apple-darwin" / "bin" / "cargo"
    )

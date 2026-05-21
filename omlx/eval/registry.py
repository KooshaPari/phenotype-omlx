# SPDX-License-Identifier: Apache-2.0
"""Benchmark registry for built-in and packaged OMLX evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkPackage:
    """Registry metadata for a benchmark package."""

    name: str
    entry_point: str
    version: str = "0.1.0"
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)


class BenchmarkRegistry:
    """Simple JSON-backed benchmark package registry."""

    def __init__(self, path: str | Path | None = None, auto_persist: bool = True):
        self.path = Path(path or Path.home() / ".omlx" / "benchmarks" / "registry.json")
        self.auto_persist = auto_persist
        self._packages: dict[str, BenchmarkPackage] = {}
        if self.auto_persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self._persist()
        if self.path.exists():
            self._packages = {
                name: BenchmarkPackage(**package)
                for name, package in self._load().get("packages", {}).items()
            }

    def register(self, package: BenchmarkPackage) -> None:
        self._packages[package.name] = package
        self._persist()

    def get(self, name: str) -> BenchmarkPackage:
        if name not in self._packages:
            raise KeyError(f"Benchmark package '{name}' is not registered")
        return self._packages[name]

    def list(self) -> list[BenchmarkPackage]:
        return list(self._packages.values())

    def search(self, query: str) -> list[BenchmarkPackage]:
        needle = query.lower()
        return [
            package
            for package in self.list()
            if needle in package.name.lower() or needle in package.description.lower()
        ]

    def remove(self, name: str) -> None:
        self._packages.pop(name, None)
        self._persist()

    def _load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _persist(self) -> None:
        if not self.auto_persist:
            return
        payload = {"packages": {name: asdict(package) for name, package in self._packages.items()}}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def builtin_registry() -> BenchmarkRegistry:
    registry = BenchmarkRegistry(auto_persist=False)
    registry._packages = {
        "swe_bench": BenchmarkPackage(
            name="swe_bench",
            entry_point="omlx.eval.swe_bench:SWEBenchBenchmark",
            description="Built-in swe_bench benchmark",
        ),
        "terminal_bench": BenchmarkPackage(
            name="terminal_bench",
            entry_point="omlx.eval.terminal_bench:TerminalBenchBenchmark",
            description="Built-in terminal_bench benchmark",
        ),
    }
    return registry
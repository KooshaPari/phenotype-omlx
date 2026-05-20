# SPDX-License-Identifier: Apache-2.0
"""Fluent builder for benchmark suites."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkStep:
    """A single benchmark entry in a suite."""

    name: str
    benchmark: str
    sample_size: int = 0
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSuite:
    """A declarative collection of benchmark steps."""

    name: str
    description: str = ""
    steps: tuple[BenchmarkStep, ...] = ()
    aggregation: str = "weighted_mean"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSuite":
        steps = tuple(BenchmarkStep(**step) for step in data.get("steps", []))
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            aggregation=data.get("aggregation", "weighted_mean"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSuite":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [asdict(step) for step in self.steps]
        return data

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def instantiate(self) -> list[tuple[BenchmarkStep, Any]]:
        return [(step, _load_benchmark(step.benchmark)(**step.params)) for step in self.steps]


class BenchmarkDSL:
    """Fluent builder for benchmark suites."""

    def __init__(self, name: str):
        self._name = name
        self._description = ""
        self._aggregation = "weighted_mean"
        self._steps: list[BenchmarkStep] = []

    def describe(self, description: str) -> "BenchmarkDSL":
        self._description = description
        return self

    def add(
        self,
        name: str,
        benchmark: str,
        sample_size: int = 0,
        weight: float = 1.0,
        **params: Any,
    ) -> "BenchmarkDSL":
        self._steps.append(BenchmarkStep(name, benchmark, sample_size, weight, params))
        return self

    def aggregate_by(self, aggregation: str) -> "BenchmarkDSL":
        self._aggregation = aggregation
        return self

    def build(self) -> BenchmarkSuite:
        return BenchmarkSuite(
            name=self._name,
            description=self._description,
            steps=tuple(self._steps),
            aggregation=self._aggregation,
        )


def _load_benchmark(path: str) -> type:
    if ":" in path:
        module_name, attr = path.split(":", 1)
        return getattr(importlib.import_module(module_name), attr)
    from . import BENCHMARKS

    if path not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{path}'")
    return BENCHMARKS[path]


def standard_suite(name: str) -> BenchmarkSuite:
    if name not in STANDARD_SUITES:
        raise KeyError(f"Unknown standard suite '{name}'")
    return STANDARD_SUITES[name]()


STANDARD_SUITES = {
    "code": lambda: BenchmarkDSL("code")
    .describe("Code generation and software engineering benchmarks")
    .add("humaneval", "humaneval")
    .add("mbpp", "mbpp")
    .add("livecodebench", "livecodebench")
    .add("swe_bench", "swe_bench", weight=2.0)
    .build(),
    "terminal": lambda: BenchmarkDSL("terminal")
    .describe("Terminal automation benchmark suite")
    .add("terminal_bench", "terminal_bench")
    .build(),
    "reasoning": lambda: BenchmarkDSL("reasoning")
    .describe("Reasoning and knowledge benchmarks")
    .add("mmlu", "mmlu")
    .add("arc_challenge", "arc_challenge")
    .add("gsm8k", "gsm8k")
    .add("truthfulqa", "truthfulqa")
    .build(),
}


def list_standard_suites() -> list[str]:
    return list(STANDARD_SUITES.keys())
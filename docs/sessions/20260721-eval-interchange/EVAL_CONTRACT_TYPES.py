"""Eval Interchange Contract v1.0 — shared types."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProducerInfo:
    name: str
    version: str
    commit_sha: str = ""


@dataclass
class RunInfo:
    run_id: str
    started_at: str
    model: str
    variant: str
    judge_mode: str = "deterministic"


@dataclass
class SuiteResult:
    suite: str
    n: int
    passed: int
    pass_at_1: float
    evidence_label: str = "reported"


@dataclass
class Totals:
    cells: int
    passed: int
    pass_at_1: float


@dataclass
class HashChain:
    top_level_sha256: str
    task_ids_sorted_sha256: str = ""


@dataclass
class EvalReport:
    contract_version: str
    producer: ProducerInfo
    run: RunInfo
    suites: list[SuiteResult]
    totals: Totals
    hash_chain: HashChain

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if self.contract_version != "0.1":
            errors.append(
                f"contract_version must be '0.1', got '{self.contract_version}'"
            )
        if not self.suites:
            errors.append("suites must not be empty")
        if self.totals.cells <= 0:
            errors.append("totals.cells must be > 0")
        if self.hash_chain.top_level_sha256 is None:
            errors.append("hash_chain.top_level_sha256 is required")
        return errors

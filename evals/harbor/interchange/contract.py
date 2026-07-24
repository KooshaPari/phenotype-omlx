"""Eval Interchange Contract v1.0 — Pydantic v2 models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProducerInfo(BaseModel):
    """Producer provenance block."""

    model_config = {"extra": "allow"}

    name: str
    version: str
    commit_sha: str = ""


class RunInfo(BaseModel):
    """Run metadata block."""

    model_config = {"extra": "allow"}

    run_id: str
    started_at: str
    model: str
    variant: str = Field(..., pattern="^(stock|ours)$")
    judge_mode: str = Field("deterministic", pattern="^(deterministic|llm|hybrid)$")


class EvidenceLabel(str, Enum):
    live_verified = "live_verified"
    reported = "reported"
    synthetic = "synthetic"


class SuiteResult(BaseModel):
    """Single suite result entry."""

    model_config = {"extra": "allow"}

    suite: str
    n: int = Field(..., ge=0)
    passed: int = Field(..., ge=0)
    pass_at_1: float = Field(..., ge=0.0, le=1.0)
    evidence_label: EvidenceLabel


class Totals(BaseModel):
    """Aggregated totals across all suites."""

    model_config = {"extra": "allow"}

    cells: int = Field(..., ge=0)
    passed: int = Field(..., ge=0)
    pass_at_1: float = Field(..., ge=0.0, le=1.0)


class HashChain(BaseModel):
    """Integrity block — hash chain for tamper detection."""

    model_config = {"extra": "forbid"}

    top_level_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    task_ids_sorted_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class EvalReport(BaseModel):
    """Top-level Eval Interchange Contract v1.0 document."""

    model_config = {"extra": "allow"}

    contract_version: str = Field(..., pattern=r"^1\.0$")
    artifact_kind: str = Field("EvaluationReport", pattern=r"^EvaluationReport$")
    producer: ProducerInfo
    run: RunInfo
    suites: list[SuiteResult] = Field(..., min_length=1)
    totals: Totals
    hash_chain: HashChain
    matrix: dict[str, Any] | None = None
    comparator: dict[str, Any] | None = None

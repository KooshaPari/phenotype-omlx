"""Eval Interchange Contract v1.0 — ingestion into eval-harness."""

from .contract import (
    EvalReport,
    HashChain,
    ProducerInfo,
    RunInfo,
    SuiteResult,
    Totals,
)
from .ingest import (
    DatasetProvenance,
    EvaluationReport,
    MultiSuiteReport,
    Suite,
    SuiteReportEntry,
    TaskResult,
    ingest_from_dict,
    ingest_from_file,
    ingest_report,
)
from .loader import load_report
from .validator import ValidationResult, validate

__all__ = [
    "DatasetProvenance",
    "EvalReport",
    "EvaluationReport",
    "HashChain",
    "MultiSuiteReport",
    "ProducerInfo",
    "RunInfo",
    "Suite",
    "SuiteReportEntry",
    "SuiteResult",
    "TaskResult",
    "Totals",
    "ValidationResult",
    "ingest_from_dict",
    "ingest_from_file",
    "ingest_report",
    "load_report",
    "validate",
]

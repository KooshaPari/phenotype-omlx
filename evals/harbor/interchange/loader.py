"""Eval Interchange Contract v1.0 — loader: JSON → validated EvalReport."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contract import EvalReport
from .validator import ValidationResult, validate

logger = logging.getLogger(__name__)


def _parse_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file."""
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def load_report(
    path: Path | str,
) -> tuple[EvalReport, ValidationResult, dict[str, Any]]:
    """Load an Eval Interchange Contract from a JSON file.

    Returns:
        (report, validation_result, raw_doc)

    Raises:
        FileNotFoundError: if the file does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
        pydantic.ValidationError: if the parsed JSON cannot build an EvalReport.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {path}")

    raw_doc = _parse_json(path)
    logger.info("Loaded contract from %s", path)

    report = EvalReport.model_validate(raw_doc)
    result = validate(report, raw_doc)

    if not result.valid:
        logger.warning(
            "Contract validation failed (%d errors): %s",
            len(result.errors),
            "; ".join(result.errors),
        )
    if result.warnings:
        for w in result.warnings:
            logger.warning(w)

    return report, result, raw_doc


def load_report_from_dict(doc: dict[str, Any]) -> tuple[EvalReport, ValidationResult]:
    """Load an Eval Interchange Contract from an already-parsed dict.

    Returns:
        (report, validation_result)
    """
    report = EvalReport.model_validate(doc)
    result = validate(report, doc)
    return report, result

# SPDX-License-Identifier: Apache-2.0
"""Interop helpers for running OMLX benchmark items from Inspect-style tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseBenchmark, QuestionResult


@dataclass(frozen=True)
class InspectSample:
    """Small Inspect-compatible sample representation."""

    id: str
    input: str
    target: str = ""
    metadata: dict[str, Any] | None = None


class InspectBridge:
    """Convert between OMLX benchmark items and Inspect-like samples."""

    def __init__(self, benchmark: BaseBenchmark):
        self.benchmark = benchmark

    def sample_from_item(self, item: dict, index: int = 0) -> InspectSample:
        messages = self.benchmark.format_prompt(item)
        prompt = "\n".join(message.get("content", "") for message in messages)
        return InspectSample(
            id=str(item.get("id", item.get("task_id", item.get("instance_id", index)))),
            input=prompt,
            target=str(item.get("answer", item.get("expected", item.get("patch", "")))),
            metadata={"benchmark": self.benchmark.name, "item": item},
        )

    def result_from_output(self, sample: InspectSample, output: str, elapsed: float = 0.0) -> QuestionResult:
        item = (sample.metadata or {}).get("item", {})
        predicted = self.benchmark.extract_answer(output, item)
        return QuestionResult(
            question_id=sample.id,
            correct=self.benchmark.check_answer(predicted, item),
            expected=sample.target,
            predicted=predicted,
            time_seconds=elapsed,
            question_text=sample.input,
            raw_response=output,
        )

    @staticmethod
    def require_inspect() -> Any:
        try:
            import inspect_ai
        except ImportError as exc:
            raise ImportError("Install inspect-ai to run native Inspect tasks") from exc
        return inspect_ai
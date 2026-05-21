# SPDX-License-Identifier: Apache-2.0
"""Terminal-Bench style benchmark adapter.

The adapter represents terminal tasks as normal OMLX benchmark questions: the
model receives a task and returns shell commands, while scoring checks command
intent without executing arbitrary model output.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl

DATA_DIR = Path(__file__).parent / "data"
DANGEROUS_SHELL_PATTERNS = (
    r"\brm\s+-rf\s+[/~]",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bchmod\s+-R\s+777\b",
)


class TerminalBenchBenchmark(BaseBenchmark):
    """Terminal task benchmark with non-executing command-sequence scoring."""

    name: str = "terminal_bench"
    quick_size: int = 50

    def __init__(self, data_path: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.data_path = data_path or str(DATA_DIR / "terminal_bench.jsonl")
        self._examples: list[dict] = []

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        if not self._examples:
            data_file = Path(self.data_path)
            if not data_file.exists():
                raise FileNotFoundError(f"Terminal-Bench dataset not found at {data_file}")
            self._examples = load_jsonl(data_file)
        if sample_size > 0:
            return deterministic_sample(self._examples, sample_size)
        return self._examples

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        setup = item.get("initial_state") or item.get("setup") or ""
        prompt = (
            "You are operating in a terminal. Return only the shell commands "
            "needed to complete the task, one command per line.\n\n"
            f"Task:\n{item.get('description', item.get('task', ''))}\n"
        )
        if setup:
            prompt += f"\nInitial state:\n{setup}\n"
        return [{"role": "user", "content": prompt}]

    def extract_answer(self, response: str, item: dict) -> str:
        blocks = re.findall(r"```(?:bash|sh|shell)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        text = blocks[-1] if blocks else response
        commands = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("$"):
                stripped = stripped[1:].strip()
            if stripped:
                commands.append(stripped)
        return "\n".join(commands)

    def check_answer(self, predicted: str, item: dict) -> bool:
        if not predicted.strip() or _has_dangerous_command(predicted):
            return False
        expected = _expected_commands(item)
        if not expected:
            return 1 <= len(predicted.splitlines()) <= 25
        predicted_tokens = _command_tokens(predicted)
        expected_tokens = set().union(*(_command_tokens(cmd) for cmd in expected))
        if not expected_tokens:
            return False
        overlap = len(predicted_tokens & expected_tokens) / len(expected_tokens)
        return overlap >= 0.5

    def get_max_tokens(self) -> int:
        return 512

    def get_question_text(self, item: dict) -> str:
        return item.get("description", item.get("task", "Terminal task"))


def _expected_commands(item: dict) -> list[str]:
    commands = item.get("commands") or item.get("expected_commands") or []
    if isinstance(commands, str):
        return [commands]
    return [str(command) for command in commands]


def _command_tokens(command: str) -> set[str]:
    return {token for token in re.split(r"[^A-Za-z0-9._/-]+", command) if token}


def _has_dangerous_command(command_text: str) -> bool:
    return any(re.search(pattern, command_text, re.IGNORECASE | re.MULTILINE) for pattern in DANGEROUS_SHELL_PATTERNS)
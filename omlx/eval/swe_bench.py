# SPDX-License-Identifier: Apache-2.0
"""SWE-Bench benchmark.

Tests software engineering ability by generating patches to fix bugs in real
GitHub issues. Evaluation requires applying the patch and running the test suite.

Dataset: SWE-Bench Lite (https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from .base import BaseBenchmark, BenchmarkResult, QuestionResult
from .datasets import deterministic_sample, load_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class SWEBenchBenchmark(BaseBenchmark):
    """SWE-Bench benchmark for software engineering tasks."""

    name: str = "swe_bench"
    quick_size: int = 50

    def __init__(self, data_path: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.data_path = data_path or str(DATA_DIR / "swe_bench_lite.jsonl")
        self._examples: list[dict] = []

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load SWE-Bench examples from JSONL file."""
        if not self._examples:
            data_file = Path(self.data_path)
            if not data_file.exists():
                raise FileNotFoundError(
                    f"SWE-Bench dataset not found at {data_file}. "
                    "Run `python -m omlx.eval.swe_bench --download` to fetch."
                )
            self._examples = load_jsonl(data_file)

        if sample_size > 0:
            return deterministic_sample(self._examples, sample_size)
        return self._examples

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format a SWE-Bench issue into a prompt for the model."""
        prompt = f"""You are an expert software engineer. Your task is to fix a bug in the repository {item['repo']}.

Problem Description:
{item['problem_statement']}

"""
        if item.get('hint_text'):
            prompt += f"Hint: {item['hint_text']}\n\n"

        prompt += """Please provide a patch that fixes the issue. The patch should be in the unified diff format.
Include the full diff starting with "diff --git" lines.

Your response should contain ONLY the patch, no additional explanation.
"""
        return [{"role": "user", "content": prompt}]

    def extract_answer(self, response: str, item: dict) -> str:
        """Extract the patch from the model response.

        Looks for content between triple backticks or assumes the entire response is the patch.
        """
        # Try to extract code block
        code_block_match = re.search(r"```(?:diff)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            return code_block_match.group(1).strip()

        # If no code block, return the stripped response
        return response.strip()

    def check_answer(self, predicted: str, item: dict) -> bool:
        """Check if the predicted patch is correct.

        For SWE-Bench, we cannot run the full test suite in this lightweight evaluator.
        Instead, we use heuristics to score the patch quality.
        """
        # Heuristic-based scoring (proxy for full evaluation)
        score = 0.0

        # Check for diff header
        if predicted.startswith("diff --git"):
            score += 0.3

        # Check for file markers
        if "+++ b/" in predicted or "--- a/" in predicted:
            score += 0.2

        # Check for hunk headers
        if re.search(r"^@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@", predicted, re.MULTILINE):
            score += 0.2

        # Check for actual changes (lines starting with + or -)
        if re.search(r"^[+-][^+-]", predicted, re.MULTILINE):
            score += 0.2

        # Reasonable length (not empty, not too large)
        lines = predicted.splitlines()
        if 5 <= len(lines) <= 500:
            score += 0.1

        return score >= 0.5

    def get_max_tokens(self) -> int:
        """SWE-Bench patches can be long."""
        return 1024

    def get_question_text(self, item: dict) -> str:
        """Return a human-readable question text."""
        return f"{item['repo']}: {item['problem_statement'][:100]}..."


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="SWE-Bench utilities")
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    args = parser.parse_args()

    if args.download:
        # Download from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
            sys.exit(1)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / "swe_bench_lite.jsonl"

        print(f"Downloading SWE-Bench Lite to {output_path}...")
        hf_hub_download(
            repo_id="princeton-nlp/SWE-bench_Lite",
            filename="swe-bench-lite-test.jsonl",
            local_dir=str(DATA_DIR),
            local_dir_use_symlinks=False,
        )
        # The downloaded file is swe-bench-lite-test.jsonl, rename to our expected name
        downloaded = DATA_DIR / "swe-bench-lite-test.jsonl"
        if downloaded.exists():
            downloaded.rename(output_path)
        print(f"Downloaded SWE-Bench Lite dataset to {output_path}")
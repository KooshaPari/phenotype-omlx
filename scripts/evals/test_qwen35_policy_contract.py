#!/usr/bin/env python3
"""Static contract tests for the Qwen3.5 policy workflow.

These tests deliberately inspect only repository files. They must never start a
model server, call Harbor, or require secrets.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github/workflows/qwen35-policy-gate.yml"
TASKS_ROOT = Path("evals/harbor/tasks")
EXPECTED_LOCAL_JOBS = {
    "evals/harbor/jobs/qwen35-policy-local.yaml",
    "evals/harbor/jobs/turboquant-ssot.yaml",
    "evals/harbor/jobs/niah-qwen35-local.yaml",
    "evals/harbor/jobs/niah-qwen35-local-32k.yaml",
}


def workflow_job_references(workflow: str) -> set[str]:
    return set(re.findall(r'^\s*YAML="([^"]+)"$', workflow, re.MULTILINE))


def task_names(job: str) -> list[str]:
    inline = re.search(r"^[ \t]*task_names:[ \t]*\[([^]]+)\]", job, re.MULTILINE)
    if inline is not None:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", inline.group(1))
    match = re.search(
        r"^[ \t]*task_names:[ \t]*\n((?:^[ \t]*-[ \t]*[A-Za-z0-9][A-Za-z0-9-]*[ \t]*\n?)+)",
        job,
        re.MULTILINE,
    )
    return [] if match is None else re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", match.group(1))


class Qwen35PolicyContractTests(unittest.TestCase):
    def test_workflow_references_are_complete_relative_qwen35_contracts(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        references = workflow_job_references(workflow)
        self.assertEqual(EXPECTED_LOCAL_JOBS, references)

        for relative_job in sorted(references):
            job_path = REPOSITORY_ROOT / relative_job
            self.assertFalse(Path(relative_job).is_absolute(), relative_job)
            self.assertTrue(job_path.is_file(), relative_job)
            job = job_path.read_text(encoding="utf-8")
            self.assertNotRegex(job, r"^\s*-\s*path:\s*/", relative_job)

            for name in task_names(job):
                task_path = REPOSITORY_ROOT / TASKS_ROOT / name
                self.assertTrue(task_path.is_dir(), f"{relative_job}: {task_path}")
                self.assertTrue((task_path / "task.toml").is_file(), f"{task_path}/task.toml")

        model_values = re.findall(
            r'^\s*(?:OPENAI_MODEL|OMLX_READY_MODEL):\s*"([^"]+)"$',
            "\n".join((REPOSITORY_ROOT / ref).read_text(encoding="utf-8") for ref in references),
            re.MULTILINE,
        )
        self.assertTrue(model_values)
        self.assertTrue(all("Qwen3.5" in value for value in model_values), model_values)

        self.assertIn("static-contract:", workflow)
        self.assertIn("if: github.event_name == 'workflow_dispatch'", workflow)

    def test_apple_niah_contract_remains_canonical(self) -> None:
        apple_job = (REPOSITORY_ROOT / "evals/harbor/jobs/niah-qwen35.yaml").read_text(encoding="utf-8")
        self.assertNotIn("DEPRECATED", apple_job)
        self.assertNotIn("superseded by", apple_job)
        self.assertIn('task_names: ["omlx-niah-api-smoke"]', apple_job)
        self.assertNotRegex(apple_job, r"^\s*-\s*path:\s*/")

    def test_validator_accepts_the_repository_contract(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/evals/validate_qwen35_policy_contract.py"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

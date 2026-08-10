#!/usr/bin/env python3
"""Validate the Qwen3.5 policy workflow without running Harbor or a model."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath


WORKFLOW = Path(".github/workflows/qwen35-policy-gate.yml")
EXPECTED_JOBS = {
    "evals/harbor/jobs/qwen35-policy-local.yaml",
    "evals/harbor/jobs/turboquant-ssot.yaml",
    "evals/harbor/jobs/niah-qwen35-local.yaml",
    "evals/harbor/jobs/niah-qwen35-local-32k.yaml",
}


class ContractError(ValueError):
    """A static workflow or Harbor contract is incomplete or unsafe."""


def repository_path(root: Path, raw_path: str, *, description: str) -> Path:
    candidate = PurePosixPath(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ContractError(f"{description} must be repository-relative: {raw_path}")
    resolved = root.joinpath(*candidate.parts)
    if not resolved.exists():
        raise ContractError(f"{description} does not exist: {raw_path}")
    return resolved


def workflow_job_references(workflow: str) -> set[str]:
    references = set(re.findall(r'^\s*YAML="([^"]+)"$', workflow, re.MULTILINE))
    if references != EXPECTED_JOBS:
        raise ContractError(
            "workflow local job references differ from the Qwen3.5 contract: "
            f"expected {sorted(EXPECTED_JOBS)}, found {sorted(references)}"
        )
    return references


def dataset_task_paths(root: Path, job_reference: str, job: str) -> list[Path]:
    if re.search(r"^[ \t]*-[ \t]*path:[ \t]*/", job, re.MULTILINE):
        raise ContractError(f"{job_reference} contains an absolute dataset path")

    datasets = list(re.finditer(
        r"^[ \t]*-[ \t]*path:[ \t]*([^\s#]+)[ \t]*\n"
        r"^[ \t]*task_names:[ \t]*\n"
        r"((?:^[ \t]*-[ \t]*[A-Za-z0-9][A-Za-z0-9-]*[ \t]*\n?)+)",
        job,
        re.MULTILINE,
    ))
    datasets.extend(
        re.finditer(
            r"^[ \t]*-[ \t]*path:[ \t]*([^\s#]+)[ \t]*\n"
            r"^[ \t]*task_names:[ \t]*\[([^]]+)\]",
            job,
            re.MULTILINE,
        )
    )
    task_paths: list[Path] = []
    for dataset in datasets:
        tasks_root = repository_path(root, dataset.group(1), description=f"{job_reference} task root")
        for task_name in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", dataset.group(2)):
            task_path = tasks_root / task_name
            if not task_path.is_dir() or not (task_path / "task.toml").is_file():
                raise ContractError(f"{job_reference} task does not exist: {task_path.relative_to(root)}")
            task_paths.append(task_path)
    if not task_paths:
        raise ContractError(f"{job_reference} defines no local task paths")
    return task_paths


def validate(root: Path) -> None:
    workflow_path = repository_path(root, WORKFLOW.as_posix(), description="workflow")
    workflow = workflow_path.read_text(encoding="utf-8")
    if "static-contract:" not in workflow:
        raise ContractError("workflow has no static contract job")
    if "if: github.event_name == 'workflow_dispatch'" not in workflow:
        raise ContractError("live Harbor/MLX job is not workflow_dispatch-only")
    if "astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e" not in workflow:
        raise ContractError("workflow does not use the verified setup-uv v6 pin")

    for reference in sorted(workflow_job_references(workflow)):
        job_path = repository_path(root, reference, description="workflow job")
        job = job_path.read_text(encoding="utf-8")
        dataset_task_paths(root, reference, job)
        models = re.findall(
            r'^[ \t]*(?:OPENAI_MODEL|OMLX_READY_MODEL):[ \t]*"([^"]+)"$', job, re.MULTILINE
        )
        if reference != "evals/harbor/jobs/turboquant-ssot.yaml" and not models:
            raise ContractError(f"{reference} declares no Qwen3.5 model")
        if any("Qwen3.5" not in model for model in models):
            raise ContractError(f"{reference} is not Qwen3.5-only: {models}")

    apple_job = (root / "evals/harbor/jobs/niah-qwen35.yaml").read_text(encoding="utf-8")
    if "DEPRECATED" in apple_job or "superseded by" in apple_job:
        raise ContractError("canonical Apple NIAH mapping is incorrectly deprecated")
    if 'task_names: ["omlx-niah-api-smoke"]' not in apple_job:
        raise ContractError("canonical Apple NIAH mapping no longer targets omlx-niah-api-smoke")
    dataset_task_paths(root, "evals/harbor/jobs/niah-qwen35.yaml", apple_job)

    niah_task = (root / "evals/harbor/tasks/omlx-niah-api-smoke/task.toml").read_text(encoding="utf-8")
    if niah_task.count('NIAH_CONTEXT_TOKENS = "${NIAH_CONTEXT_TOKENS:-0}"') != 3:
        raise ContractError("8k NIAH context must reach environment, solution, and agent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root to validate")
    args = parser.parse_args()
    try:
        validate(args.root.resolve())
    except ContractError as error:
        print(f"qwen35 static contract: {error}", file=sys.stderr)
        return 1
    print("qwen35 static contract: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

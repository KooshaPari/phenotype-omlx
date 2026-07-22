"""Immutable checked-in workload binding for real-model evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

if "e2e_validation" in sys.modules:
    validation = sys.modules["e2e_validation"]
elif "scripts.e2e_validation" in sys.modules:
    validation = sys.modules["scripts.e2e_validation"]
else:
    try:
        import e2e_validation as validation
    except ModuleNotFoundError:
        from scripts import e2e_validation as validation


@dataclass(frozen=True)
class BenchmarkWorkload:
    """A checked-in workload, identified by the digest of its exact bytes."""

    name: str
    kind: str
    prompt: str
    teacher_forced_continuation: str
    revision: str


BENCHMARK_WORKLOAD_PATH = (
    Path(__file__).with_name("workloads") / "fibonacci-teacher-forced-v1.json"
)


def load_benchmark_workload(path: Path = BENCHMARK_WORKLOAD_PATH) -> BenchmarkWorkload:
    """Load one local workload and bind its immutable content digest."""

    payload = Path(path).read_bytes()
    try:
        document = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise validation.ValidationError("benchmark workload must be valid JSON") from error
    required = ("name", "kind", "prompt", "teacher_forced_continuation")
    if not isinstance(document, dict) or any(
        not isinstance(document.get(field), str) or not document[field]
        for field in required
    ):
        raise validation.ValidationError("benchmark workload fields must be nonempty strings")
    if document.get("schema_version") != 1:
        raise validation.ValidationError("benchmark workload schema_version must be 1")
    if document["kind"] != "checked_in_benchmark_workload":
        raise validation.ValidationError("benchmark workload kind is not supported")
    return BenchmarkWorkload(
        name=document["name"],
        kind=document["kind"],
        prompt=document["prompt"],
        teacher_forced_continuation=document["teacher_forced_continuation"],
        revision=hashlib.sha256(payload).hexdigest(),
    )

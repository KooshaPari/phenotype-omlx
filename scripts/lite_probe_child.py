#!/usr/bin/env python3
"""Isolated, non-publishing entrypoint for a future Lite cache capability probe."""

from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    from e2e_real_model_host import load_benchmark_workload
    from e2e_validation import ValidationError, ValidationManifest
except ModuleNotFoundError:
    from scripts.e2e_real_model_host import load_benchmark_workload
    from scripts.e2e_validation import ValidationError, ValidationManifest


def parse_request(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValidationError("probe request must be an object")
    required = ("model_path", "model_revision", "tokenizer_revision", "workload_path", "workload_revision")
    if any(not isinstance(payload.get(name), str) or not payload[name] for name in required):
        raise ValidationError("probe request fields must be nonempty strings")
    workload = load_benchmark_workload(Path(payload["workload_path"]))
    model_path = Path(payload["model_path"]).resolve()
    snapshot_revision = payload["model_revision"].removeprefix("git:")
    if not model_path.is_dir() or model_path.name != snapshot_revision:
        raise ValidationError("model_path must be the resolved immutable snapshot directory")
    if payload["workload_revision"] != workload.revision:
        raise ValidationError("workload_revision must match exact workload bytes")
    ValidationManifest.create(
        model_revision=payload["model_revision"],
        corpus_revision=payload["workload_revision"],
        tokenizer_revision=payload["tokenizer_revision"],
    )
    result = {name: payload[name] for name in required}
    result["model_path"] = str(model_path)
    result["workload_path"] = str(Path(payload["workload_path"]).resolve())
    return result


def actual_model_probe(_request: dict[str, str]) -> dict[str, object]:
    """Reserved seam for pinned MLX work; intentionally disabled until GO."""

    raise RuntimeError("actual Lite probe is disabled pending an uncontended Metal window")


def main() -> int:
    try:
        request = parse_request(json.load(sys.stdin))
        # Do not call actual_model_probe here: validation mode is deliberately inert.
        print(json.dumps({
            "status": "capability_pending",
            "publication": False,
            "model_revision": request["model_revision"],
            "workload_revision": request["workload_revision"],
        }))
        return 0
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "invalid_request", "publication": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

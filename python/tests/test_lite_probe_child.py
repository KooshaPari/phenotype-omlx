"""Subprocess contract for the isolated Lite capability-probe child."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CHILD = ROOT / "scripts" / "lite_probe_child.py"


def test_child_validates_immutable_request_without_running_model() -> None:
    request = {
        "model_path": "/Users/kooshapari/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
        "model_revision": "git:a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
        "tokenizer_revision": "b" * 64,
        "workload_path": str(ROOT / "scripts" / "workloads" / "fibonacci-teacher-forced-v1.json"),
        "workload_revision": "0edd5cab55ad65d7a4e471df507e2a4426a492f19bb47a4a00d000492a5c3e66",
    }
    completed = subprocess.run(
        [sys.executable, str(CHILD)], input=json.dumps(request), text=True, capture_output=True, check=True
    )
    response = json.loads(completed.stdout)
    assert response["status"] == "capability_pending"
    assert response["publication"] is False

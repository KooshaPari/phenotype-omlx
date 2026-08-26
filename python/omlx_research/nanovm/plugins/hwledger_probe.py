"""hwLedger probe plugin — inventory snapshot for control-plane federation.

Does not run inference. Emits a JSON heartbeat suitable for
`pheno.device.heartbeat` (NATS) or local file drop.
"""

from __future__ import annotations

import json
import csv
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


def _nvidia_smi() -> list[dict[str, str]]:
    candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for bin_path in candidates:
        try:
            out = subprocess.check_output(
                [
                    bin_path,
                    "--query-gpu=name,uuid,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=8,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        gpus = []
        for parts in csv.reader(out.splitlines(), skipinitialspace=True):
            parts = [p.strip() for p in parts]
            if len(parts) >= 4:
                gpus.append(
                    {
                        "name": parts[0],
                        "uuid": parts[1],
                        "memory_total": parts[2],
                        "driver": parts[3],
                    }
                )
        if gpus:
            return gpus
    return []


def snapshot() -> dict[str, Any]:
    return {
        "schema": "pheno.device.heartbeat/v0",
        "ts": time.time(),
        "host": socket.gethostname(),
        "platform": platform.system().lower(),
        "machine": platform.machine(),
        "gpus": _nvidia_smi(),
        "source": "hwledger-probe",
        "omlx_root": os.environ.get("OMLX_ROOT", ""),
        "hwledger_root": os.environ.get("HWLEDGER_ROOT", ""),
    }


def publish_local(path: str | Path | None = None) -> Path:
    """Publish one heartbeat without overwriting an existing evidence file."""
    out = Path(
        path
        or os.environ.get(
            "HWLEDGER_HEARTBEAT_PATH", "platform/federation/out/heartbeat.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    data = snapshot()
    payload = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    with out.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return out


if __name__ == "__main__":
    print(publish_local())

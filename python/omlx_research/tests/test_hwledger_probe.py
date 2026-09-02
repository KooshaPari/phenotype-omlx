"""Contract tests for the hwLedger NanoVM heartbeat probe."""

from __future__ import annotations

import json
from pathlib import Path

from omlx_research.nanovm.plugins import hwledger_probe


def test_nvidia_smi_parses_quoted_commas(monkeypatch) -> None:
    monkeypatch.setattr(
        hwledger_probe.subprocess,
        "check_output",
        lambda *args, **kwargs: '"NVIDIA, Test", GPU-uuid, 4096 MiB, 535.1\n',
    )
    assert hwledger_probe._nvidia_smi() == [
        {
            "name": "NVIDIA, Test",
            "uuid": "GPU-uuid",
            "memory_total": "4096 MiB",
            "driver": "535.1",
        }
    ]


def test_snapshot_is_inventory_only(monkeypatch) -> None:
    monkeypatch.setattr(hwledger_probe, "_nvidia_smi", list)
    report = hwledger_probe.snapshot()
    assert report["schema"] == "pheno.device.heartbeat/v0"
    assert report["source"] == "hwledger-probe"
    assert report["gpus"] == []
    assert "inference" not in report


def test_publish_local_writes_valid_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hwledger_probe, "_nvidia_smi", list)
    output = hwledger_probe.publish_local(tmp_path / "heartbeat.json")
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == (
        "pheno.device.heartbeat/v0"
    )


def test_publish_local_refuses_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hwledger_probe, "_nvidia_smi", list)
    output = tmp_path / "heartbeat.json"
    output.write_text("preserve-me", encoding="utf-8")
    try:
        hwledger_probe.publish_local(output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("heartbeat publication must not overwrite evidence")
    assert output.read_text(encoding="utf-8") == "preserve-me"

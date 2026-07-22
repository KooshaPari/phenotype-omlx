"""FR-7 D0–D2 unit tests for the canonical oMLX vPU dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from omlx_research.cli._doctor_shared import project_root
from omlx_research.vpu_dashboard.server import build_status, health_payload, panel_index, schema_path


def test_fr7_contract_and_assets_present():
    root = Path(project_root())
    assert (root / "perf-core" / "vpu" / "dashboard" / "CONTRACT.md").is_file()
    assert schema_path().is_file()
    assert panel_index().is_file()
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    assert schema["title"].startswith("OMLX vPU Dashboard Status")


def test_fr7_health_and_status_shape():
    body, code = health_payload()
    assert code == 200
    assert body["ok"] is True
    assert body["owner"] == "Salmon"

    status = build_status()
    for key in (
        "schema_version",
        "build_head",
        "polyglot_tiers",
        "eval_snapshot_id",
        "promotion_snapshot_id",
        "errors",
        "owner",
    ):
        assert key in status
    assert status["schema_version"] == 1
    assert status["owner"] == "Salmon"
    assert status["errors"] == []

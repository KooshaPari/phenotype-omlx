"""Structural boundaries for readiness orchestration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_readiness_orchestrator_stays_within_the_350_line_target() -> None:
    """Keep command and wheel mechanics outside top-level readiness policy."""

    orchestrator = ROOT / "scripts" / "readiness_check.py"
    assert len(orchestrator.read_text().splitlines()) <= 350

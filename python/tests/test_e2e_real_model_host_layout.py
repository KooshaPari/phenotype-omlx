"""Structural boundaries for the real-model host adapter."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_host_boundary_stays_within_the_350_line_target() -> None:
    """Keep probe and workload concerns outside the host evidence boundary."""

    host = ROOT / "scripts" / "e2e_real_model_host.py"
    assert len(host.read_text().splitlines()) <= 350

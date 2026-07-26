"""Contract test for immutable, non-synthetic Qwen3.5 FR-5 E3 evidence."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evals.verify_fr5_e3_artifact import verify  # noqa: E402


def test_promoted_e3_evidence_is_hashed_and_does_not_promote_local_e4() -> None:
    verify()

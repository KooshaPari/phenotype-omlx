"""Unit tests for L2 contamination scan helpers."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "evals"
        / "contamination_scan.py"
    )
    spec = importlib.util.spec_from_file_location("contamination_scan", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestContaminationScan(unittest.TestCase):
    def test_jaccard_identical(self) -> None:
        """FR-L2-001: identical n-gram multisets → jaccard 1."""
        mod = _load()
        c = mod._ngrams(mod._tokens("alpha beta gamma delta epsilon"), 3)
        self.assertEqual(mod.jaccard(c, c), 1.0)

    def test_degenerate_synthetic_verdict(self) -> None:
        """FR-L2-002: all pass@1=1 and judge=0 → UNTRUSTED_SYNTHETIC."""
        mod = _load()
        cells = [
            {"pass_at_1": 1.0, "judge_score": 0.0, "reply": "ok", "task_id": "t1"}
            for _ in range(3)
        ]
        result = mod.scan(cells, fixtures=[])
        self.assertEqual(result["verdict"], "UNTRUSTED_SYNTHETIC")


if __name__ == "__main__":
    unittest.main()

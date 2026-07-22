"""Unit test for L6 RLVR-AF smoke contract."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[3] / "scripts" / "evals" / "rlvr_af_smoke.py"
    spec = importlib.util.spec_from_file_location("rlvr_af_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestRlvrAfSmoke(unittest.TestCase):
    def test_hard_verifier_contract(self) -> None:
        """FR-L6-001: needle/schema pass; empty/error fail."""
        mod = _load()
        result = mod.run_smoke()
        self.assertTrue(result["contract_ok"], result)
        self.assertEqual(result["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()

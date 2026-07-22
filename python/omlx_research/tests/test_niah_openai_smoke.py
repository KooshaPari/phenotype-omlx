"""Unit tests for NIAH OpenAI smoke (no network)."""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[3] / "scripts" / "evals" / "niah_openai_smoke.py"
    spec = importlib.util.spec_from_file_location("niah_openai_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestNiahOpenaiSmoke(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OMLX_READY_MODEL", None)
        os.environ.pop("OPENAI_MODEL", None)

    def test_requires_base_url(self) -> None:
        """FR-NIAH-API-001: fail loud without OPENAI_BASE_URL."""
        mod = _load()
        with self.assertRaises(SystemExit) as ctx:
            mod.run_niah()
        self.assertIn("OPENAI_BASE_URL", str(ctx.exception))

    def test_rejects_qwen25_model(self) -> None:
        """FR-NIAH-API-002: Qwen2.5 quarantined."""
        mod = _load()
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
        os.environ["OMLX_READY_MODEL"] = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        with self.assertRaises(SystemExit) as ctx:
            mod.run_niah()
        self.assertIn("Qwen2.5", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

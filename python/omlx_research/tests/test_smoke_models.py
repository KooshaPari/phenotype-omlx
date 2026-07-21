"""FR-TEST: smoke_models SSOT — Qwen3.5 gate + role defaults."""
from __future__ import annotations

import os
import unittest

from omlx_research.smoke_models import (
    SmokeModelError,
    assert_qwen35,
    default_model_for,
    load_smoke_config,
)


class TestSmokeModels(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("OMLX_READY_MODEL", None)
        os.environ.pop("OMLX_ALLOW_LEGACY_QWEN25", None)
        load_smoke_config.cache_clear()

    def test_defaults_are_qwen35(self) -> None:
        """FR-SMOKE-001: SSOT defaults contain Qwen3.5."""
        cfg = load_smoke_config()
        for key, val in (cfg.get("defaults") or {}).items():
            self.assertIn("Qwen3.5", str(val), msg=key)

    def test_readiness_role(self) -> None:
        """FR-SMOKE-002: readiness role resolves OptiQ Qwen3.5."""
        m = default_model_for("readiness")
        self.assertIn("Qwen3.5", m)

    def test_rejects_qwen25(self) -> None:
        """FR-SMOKE-003: Qwen2.5 refused without escape hatch."""
        with self.assertRaises(SmokeModelError):
            assert_qwen35("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    def test_legacy_escape(self) -> None:
        """FR-SMOKE-004: OMLX_ALLOW_LEGACY_QWEN25 opts into quarantine."""
        os.environ["OMLX_ALLOW_LEGACY_QWEN25"] = "1"
        m = assert_qwen35("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        self.assertIn("Qwen2.5", m)


if __name__ == "__main__":
    unittest.main()

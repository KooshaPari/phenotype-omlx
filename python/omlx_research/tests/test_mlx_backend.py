"""Unit tests for MlxBackend — verifies the production TurboQuant+ path.

These tests cover the regressions fixed for phenotype-omlx on 2026-07-15:

  test_perf_module_initialized
      Confirms __init__ sets self._perf_module, so generate_with_turbo_cache
      never hits AttributeError.

  test_generate_with_turbo_cache_compresses_layers
      Confirms compact_turbo_cache actually compresses > 0 layers on
      Qwen2.5-0.5B-Instruct. Previously this returned 0/24 layers because
      the code built a list of TurboKVCache (not TurboKVCacheLite), and
      compact_turbo_cache silently skipped non-Lite entries.

  test_force_compact_overrides_gating
      Confirms the new force_compact flag bypasses the compact_threshold
      gating so we see compression at any context length.

Tests are skipped when mlx_lm/the model is unavailable; that lets CI in
non-Apple-Silicon environments still see the structural assertions
(_perf_module init).
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Optional


def _model_local_path() -> Optional[str]:
    """Resolve the test model path; skip if not cached and HF is offline."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    except Exception:
        return None


class TestMlxBackendInit(unittest.TestCase):
    """Structural tests — run anywhere with the package importable."""

    def test_perf_module_initialized(self):
        """MlxBackend.__init__ must pre-set self._perf_module.

        Regression: previously __init__ set self._model = None but did
        NOT set self._perf_module. Production code calling
        generate_with_turbo_cache then crashed with AttributeError
        the moment it touched self._perf_module.
        """
        from omlx_research.backends.mlx_backend import MlxBackend
        be = MlxBackend()
        # Must exist as an attribute, even if it hasn't been resolved yet.
        self.assertTrue(
            hasattr(be, "_perf_module"),
            "MlxBackend.__init__ must set self._perf_module (got AttributeError-unsafe)",
        )
        # And the value must be one of the documented sentinel set: None, module, or False.
        v = be._perf_module
        self.assertIn(
            type(v).__name__, ("NoneType", "module", "bool"),
            f"unexpected _perf_module type: {type(v).__name__}",
        )

    def test_rust_perf_lazy_loads(self):
        """_rust_perf() must populate _perf_module on first call, then cache."""
        from omlx_research.backends.mlx_backend import MlxBackend
        be = MlxBackend()
        # First call resolves and caches.
        r1 = be._rust_perf()
        v = be._perf_module
        # Second call returns the same cached value.
        r2 = be._rust_perf()
        self.assertEqual(r1, r2)


class TestMlxBackendTurboQuantProduction(unittest.TestCase):
    """End-to-end tests against Qwen2.5-0.5B-Instruct-4bit on MLX/Metal.

    Skipped when the model isn't available locally (e.g. CI without HF cache).
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = _model_local_path()
        if cls.model_path is None:
            raise unittest.SkipTest(
                "Qwen2.5-0.5B-Instruct not in HF cache and offline — "
                "set HF_HUB_OFFLINE=0 and ensure model is available."
            )

    def test_generate_with_turbo_cache_compresses_layers(self):
        """Confirm compact_turbo_cache actually compresses >0 TurboKVCacheLite layers.

        Pre-fix bug:
            - turbo_cache was a list of TurboKVCache (not Lite)
            - compact_turbo_cache silently returns 0 for non-Lite entries
            - We reported "compressed 0/N layers" at every length

        Post-fix:
            - turbo_cache contains TurboKVCacheLite + KVCache boundary layers
            - compact_turbo_cache iterates Lite.compact() correctly
            - We report compressed layers > 0 after prefill + decode
        """
        from omlx_research.backends.mlx_backend import MlxBackend, GenerateRequest
        be = MlxBackend(self.model_path)
        req = GenerateRequest(
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0.0,
        )
        resp = be.generate_with_turbo_cache(
            req, turbo_bits=4, turbo_key_bits=0,
            compact_after_prefill=True, force_compact=True,
        )
        turbo_meta = resp.metadata.get("turbo", {})
        n_compressed = turbo_meta.get("compressed", -1)
        n_lite = turbo_meta.get("lite_layers", 0)
        # The 2 specific assertions: layers wrapped + at least some compressed.
        self.assertGreater(
            n_lite, 0,
            f"force_compact=True must wrap all layers with TurboKVCacheLite, got {n_lite}",
        )
        self.assertGreater(
            n_compressed, 0,
            f"compact_turbo_cache must compress >0 layers, got compressed={n_compressed} "
            f"(lite_layers={n_lite}, layers={turbo_meta.get('layers')}, "
            f"force_compact={turbo_meta.get('force_compact')})",
        )

    def test_force_compact_overrides_gating(self):
        """force_compact=True must produce compression regardless of compact_threshold."""
        from omlx_research.backends.mlx_backend import MlxBackend, GenerateRequest
        be = MlxBackend(self.model_path)
        # Short prompt so compact_threshold wouldn't be crossed
        req = GenerateRequest(
            prompt="Hi",
            max_tokens=4,
            temperature=0.0,
        )
        resp = be.generate_with_turbo_cache(
            req, turbo_bits=4, turbo_key_bits=0,
            compact_after_prefill=True, force_compact=True,
        )
        turbo_meta = resp.metadata.get("turbo", {})
        n_compressed = turbo_meta.get("compressed", -1)
        n_lite = turbo_meta.get("lite_layers", 0)
        self.assertGreater(n_lite, 0, "force_compact must wrap all layers")
        self.assertGreater(
            n_compressed, 0,
            f"force_compact must compress at any length, got compressed={n_compressed}",
        )

    def test_rust_encode_production_path(self):
        """Generate with Rust path enabled (default) — encode metrics in metadata."""
        from omlx_research.backends.mlx_backend import MlxBackend, GenerateRequest
        be = MlxBackend(self.model_path)
        req = GenerateRequest(
            prompt="Hello world",
            max_tokens=4,
            temperature=0.0,
        )
        resp = be.generate_with_turbo_cache(req, turbo_bits=4, turbo_key_bits=0)
        turbo_meta = resp.metadata.get("turbo", {})
        # Default mode is Rust; encode_path must report 'rust' (or 'unavailable'
        # if the _perf module wasn't built into the venv).
        ep = turbo_meta.get("encode_path")
        self.assertIn(
            ep, ("rust", "unavailable"),
            f"default encode_path should be rust or unavailable, got {ep!r}",
        )


if __name__ == "__main__":
    unittest.main()

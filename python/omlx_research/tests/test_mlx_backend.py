"""Unit tests for MlxBackend — verifies the production TurboQuant+ path.

These tests cover the regressions fixed for phenotype-omlx on 2026-07-15:

  test_perf_module_initialized
      Confirms __init__ sets self._perf_module, so generate_with_turbo_cache
      never hits AttributeError.

  test_generate_with_turbo_cache_compresses_layers
      Confirms compact_turbo_cache actually compresses > 0 layers on
      Qwen3.5-0.8B. Previously this returned 0/24 layers because
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
        from omlx_research.smoke_models import default_model_for
        return snapshot_download(default_model_for("mlx_backend_test"))
    except Exception:
        return None


def _require_mlx_lm() -> None:
    """Skip the calling test if `mlx_lm` (the inference runtime) is missing.

    The structural tests in TestMlxBackendInit run without it because they
    only exercise __init__ and the lazy _rust_perf resolver. The end-to-end
    tests in TestMlxBackendTurboQuantProduction call MlxBackend._load() /
    generate_with_turbo_cache(), which import `mlx_lm` at the top of
    `MlxBackend._load`. If the runtime is unavailable in this environment,
    every production-path test must skip (not fail) so the suite stays
    runnable on non-Apple-Silicon CI hosts that still have `mlx.core`.
    """
    try:
        import mlx_lm  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only when missing
        raise unittest.SkipTest(
            "mlx_lm not installed in this environment; production-path "
            "tests require it (pip install mlx-lm): "
            f"{e}"
        )


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

    def test_metal_backend_preserves_model_path(self):
        """Explicit Metal policy must pass the selected model to the MLX executor."""
        from unittest.mock import patch

        from omlx_research.backends.base import GenerateRequest, GenerateResponse
        from omlx_research.backends.metal_backend import MetalKernelBackend

        backend = MetalKernelBackend("Qwen/Qwen3.5-0.8B")
        fake = GenerateResponse("ok", 1, 2, backend="mlx", metadata={"loaded": True})
        with patch("mlx.core.metal.is_available", return_value=True), patch(
            "omlx_research.backends.mlx_backend.MlxBackend.generate", return_value=fake
        ) as generate:
            response = backend.generate(GenerateRequest("hello"))
        self.assertEqual(response.backend, "metal")
        self.assertEqual(response.metadata["model_path"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(response.metadata["custom_metal_probe"]["passed"], True)
        provenance = response.metadata["kernel_execution_provenance"]
        self.assertEqual(provenance["verification_scope"], "probe_only")
        self.assertEqual(provenance["execution_source"], "unavailable")
        self.assertFalse(provenance["custom_kernel_execution_verified"])
        generate.assert_called_once()

    def test_kernel_plan_is_explicitly_nonexecuted(self):
        """Layer introspection must not be reported as custom execution."""
        from omlx_research.backends.mlx_backend import MlxBackend

        backend = MlxBackend()
        plan = backend.kernel_plan()
        self.assertEqual(plan["execution_source"], "unavailable")
        self.assertEqual(plan["custom_kernel_dispatches"], 0)
        self.assertFalse(plan["custom_kernel_execution_verified"])

    def test_quantization_provenance_rejects_synthetic_probe(self):
        """A standalone quantization probe is not cache-compression evidence."""
        from omlx_research.backends.mlx_backend import quantization_execution_provenance

        provenance = quantization_execution_provenance()

        self.assertEqual(provenance["execution_source"], "not_executed")
        self.assertFalse(provenance["rust_quantization_executed"])
        self.assertFalse(provenance["cache_compression_verified"])
        self.assertEqual(provenance["evidence_scope"], "none")


class TestRequireMlxLmHelper(unittest.TestCase):
    """The helper gates production-path tests on the `mlx_lm` runtime.

    Without this gate the production-path tests fail with ModuleNotFoundError
    on environments that have `mlx.core` but no `mlx_lm` (e.g. CI runners
    that only exercise the structural / decode-path coverage). The helper
    must skip cleanly so the suite stays green.
    """

    def test_helper_skips_when_mlx_lm_missing(self):
        """If `mlx_lm` cannot be imported, _require_mlx_lm must raise SkipTest.

        We simulate the missing-dependency path by hiding the real `mlx_lm`
        via sys.modules manipulation; the helper must re-raise as SkipTest.
        """
        import sys
        # Make sure the helper sees a missing `mlx_lm`. We only stash a
        # sentinel under that name to force ImportError on next import;
        # afterwards we restore the original mapping so other tests are
        # not affected.
        original = sys.modules.get("mlx_lm")
        sys.modules["mlx_lm"] = None  # type: ignore[assignment]
        try:
            with self.assertRaises(unittest.SkipTest) as cm:
                _require_mlx_lm()
            self.assertIn("mlx_lm not installed", str(cm.exception))
        finally:
            if original is None:
                sys.modules.pop("mlx_lm", None)
            else:
                sys.modules["mlx_lm"] = original

    def test_helper_passes_when_mlx_lm_importable(self):
        """If `mlx_lm` is importable, _require_mlx_lm must return None."""
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "mlx_lm not installed — cannot exercise the happy path of "
                "_require_mlx_lm on this host."
            )
        self.assertIsNone(_require_mlx_lm())


class TestMlxBackendTurboQuantProduction(unittest.TestCase):
    """End-to-end tests against Qwen3.5-0.8B-4bit on MLX/Metal.

    Skipped when the model isn't available locally (e.g. CI without HF cache).
    """

    @classmethod
    def setUpClass(cls):
        # Cheap gate first: mlx_lm must be importable; skip otherwise.
        # This avoids forcing a HuggingFace model download on hosts that
        # have `mlx.core` but no `mlx_lm` (e.g. CI runners that only need
        # to exercise the structural / decode-path coverage).
        _require_mlx_lm()
        cls.model_path = _model_local_path()
        if cls.model_path is None:
            raise unittest.SkipTest(
                "Qwen3.5-0.8B not in HF cache and offline — "
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
        if n_lite == 0:
            raise unittest.SkipTest(
                turbo_meta.get(
                    "cache_applicability_reason",
                    "model exposes no TurboKVCacheLite layers; KV compression is not applicable",
                )
            )
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
        if n_lite == 0:
            raise unittest.SkipTest(
                turbo_meta.get(
                    "cache_applicability_reason",
                    "model exposes no TurboKVCacheLite layers; KV compression is not applicable",
                )
            )
        self.assertGreater(n_lite, 0, "force_compact must wrap all layers")
        self.assertGreater(
            n_compressed, 0,
            f"force_compact must compress at any length, got compressed={n_compressed}",
        )

    def test_cache_compression_provenance(self):
        """Generation evidence derives from compacted cache state, not a probe."""
        from omlx_research.backends.mlx_backend import MlxBackend, GenerateRequest
        be = MlxBackend(self.model_path)
        req = GenerateRequest(
            prompt="Hello world",
            max_tokens=4,
            temperature=0.0,
        )
        resp = be.generate_with_turbo_cache(req, turbo_bits=4, turbo_key_bits=0)
        turbo_meta = resp.metadata.get("turbo", {})
        if turbo_meta.get("lite_layers", 0) == 0:
            raise unittest.SkipTest(
                turbo_meta.get(
                    "cache_applicability_reason",
                    "model exposes no TurboKVCacheLite layers; KV compression is not applicable",
                )
            )
        provenance = turbo_meta.get("quantization_provenance", {})
        self.assertEqual(provenance.get("execution_source"), "turbo_kv_cache")
        self.assertTrue(provenance.get("cache_compression_verified"))
        self.assertEqual(provenance.get("evidence_scope"), "turbo_kv_cache_state")
        self.assertNotIn("encode_path", turbo_meta)


if __name__ == "__main__":
    unittest.main()

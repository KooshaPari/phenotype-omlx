"""Integration tests for MlxBackend — error handling and load failure paths.

These tests exercise the failure modes of MlxBackend._load() and
generate() when the model cannot be loaded. They use mocking to simulate
loading failures without requiring MLX or a real model.

FR-XXX-001: MlxBackend must surface load errors via _load_error attribute.
FR-XXX-002: MlxBackend.generate() must return an error response when model is not loaded.
FR-XXX-003: MlxBackend._load() must log a warning when loading fails.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from omlx_research.backends.mlx_backend import MlxBackend
from omlx_research.backends.base import GenerateRequest


class TestMlxBackendLoadFailure(unittest.TestCase):
    """Test _load() error storage and logging when model loading fails."""

    def test_load_stores_error_on_failure(self):
        """_load() must store the exception message in _load_error on failure."""
        be = MlxBackend(model_path="/nonexistent/model")
        self.assertIsNone(be._load_error)

        mock_mlx = MagicMock()
        mock_mlx.load.side_effect = RuntimeError("Model not found")

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx}):
            be._load()

        self.assertIsNotNone(be._load_error)
        self.assertIn("Model not found", be._load_error)

    def test_load_error_attribute_defaults_to_none(self):
        """After construction, _load_error must be None (no prior attempt)."""
        be = MlxBackend(model_path="/some/path")
        self.assertIsNone(be._load_error)

    def test_load_error_message_format(self):
        """The _load_error message must include the original exception string."""
        be = MlxBackend(model_path="/bad/path")

        mock_mlx = MagicMock()
        mock_mlx.load.side_effect = OSError("Permission denied: /bad/path")

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx}):
            be._load()

        self.assertIsNotNone(be._load_error)
        self.assertIn("Permission denied", be._load_error)
        self.assertIn("/bad/path", be._load_error)

    def test_generate_returns_error_response_when_model_not_loaded(self):
        """generate() must return a GenerateResponse with error when model is None."""
        be = MlxBackend(model_path=None)
        req = GenerateRequest(prompt="Hello", max_tokens=10)
        resp = be.generate(req)
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.tokens, 0)
        self.assertEqual(resp.elapsed_ms, 0)
        self.assertEqual(resp.backend, "mlx")
        self.assertIn("error", resp.metadata)

    def test_generate_with_turbo_cache_returns_error_when_model_not_loaded(self):
        """generate_with_turbo_cache() must return error response when model is None."""
        be = MlxBackend(model_path=None)
        req = GenerateRequest(prompt="Hello", max_tokens=10)
        resp = be.generate_with_turbo_cache(req)
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.tokens, 0)
        self.assertEqual(resp.elapsed_ms, 0)
        self.assertEqual(resp.backend, "mlx")
        self.assertIn("error", resp.metadata)

    def test_generate_returns_error_after_failed_load(self):
        """generate() must return error response after _load() has failed."""
        be = MlxBackend(model_path="/fail/model")

        mock_mlx = MagicMock()
        mock_mlx.load.side_effect = RuntimeError("CUDA OOM")

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx}):
            be._load()

        req = GenerateRequest(prompt="Test", max_tokens=5)
        resp = be.generate(req)
        self.assertEqual(resp.text, "")
        self.assertIn("error", resp.metadata)


class TestMlxBackendLoadLogging(unittest.TestCase):
    """Test that _load() emits a warning on failure."""

    def test_load_logs_warning_on_failure(self):
        """_load() must log a warning when mlx_lm.load raises."""
        be = MlxBackend(model_path="/nonexistent/model")

        mock_mlx = MagicMock()
        mock_mlx.load.side_effect = RuntimeError("CUDA OOM")

        with (
            patch.dict(sys.modules, {"mlx_lm": mock_mlx}),
            patch("omlx_research.backends.mlx_backend.logger") as mock_logger,
        ):
            be._load()

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        log_msg = str(call_args)
        self.assertIn("CUDA OOM", log_msg)

    def test_load_no_warning_on_success(self):
        """_load() must NOT log a warning when loading succeeds."""
        be = MlxBackend(model_path="/good/model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.load.return_value = (mock_model, mock_tokenizer)

        with (
            patch.dict(sys.modules, {"mlx_lm": mock_mlx}),
            patch("omlx_research.backends.mlx_backend.logger") as mock_logger,
        ):
            be._load()

        mock_logger.warning.assert_not_called()
        self.assertIs(be._model, mock_model)


class TestMlxBackendLoadIdempotent(unittest.TestCase):
    """Test that _load() does not retry after first failure."""

    def test_load_does_not_retry_after_failure(self):
        """Second _load() call must not call mlx_lm.load again if _load_error is set."""
        be = MlxBackend(model_path="/fail/model")
        call_count = 0

        def failing_load(path):
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"Attempt {call_count}")

        mock_mlx = MagicMock()
        mock_mlx.load.side_effect = failing_load

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx}):
            be._load()
            be._load()

        self.assertEqual(call_count, 1, "_load() must not retry after first failure")

    def test_load_success_does_not_set_error(self):
        """_load_error must remain None after a successful load."""
        be = MlxBackend(model_path="/ok/model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx = MagicMock()
        mock_mlx.load.return_value = (mock_model, mock_tokenizer)

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx}):
            be._load()

        self.assertIsNone(be._load_error)


if __name__ == "__main__":
    unittest.main()

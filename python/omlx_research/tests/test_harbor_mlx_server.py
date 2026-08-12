"""Regression coverage for the dedicated Harbor Qwen3.5 MLX server adapter."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from omlx_research.harbor_mlx_server import (
    HarborMlxServerError,
    install_worker_generation_stream,
    patch_mlx_lm_generation_worker,
    server_command,
)


class _FakeMlx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def default_device(self) -> str:
        self.calls.append(("default_device", None))
        return "gpu"

    def new_thread_local_stream(self, device: str) -> str:
        self.calls.append(("new_thread_local_stream", device))
        return f"thread-local:{device}"


class TestHarborMlxServer(unittest.TestCase):
    def test_niah_operator_hint_uses_thread_safe_adapter(self) -> None:
        """The no-endpoint path directs Harbor to the dedicated adapter, not vanilla mlx-lm."""
        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / "scripts" / "evals" / "run_via_harbor.sh"
        source = script.read_text(encoding="utf-8")

        self.assertIn("python3 -m omlx_research.harbor_mlx_server", source)
        self.assertIn("--port 8766", source)
        self.assertNotIn("mlx_lm server --model", source)

    def test_worker_rebinds_generation_stream_with_thread_local_mlx_stream(self) -> None:
        """FR-5: the MLX generation worker owns the stream it evaluates on."""
        mlx = _FakeMlx()
        generate = types.SimpleNamespace(generation_stream="import-thread-stream")
        server = types.SimpleNamespace(generation_stream="import-thread-stream")

        stream = install_worker_generation_stream(mlx, generate, server)

        self.assertEqual("thread-local:gpu", stream)
        self.assertEqual(stream, generate.generation_stream)
        self.assertEqual(stream, server.generation_stream)
        self.assertEqual(
            [("default_device", None), ("new_thread_local_stream", "gpu")], mlx.calls
        )

    def test_patch_installs_stream_in_generation_worker_not_import_thread(self) -> None:
        """The adapter patches only the dedicated mlx-lm generation worker."""
        mlx = _FakeMlx()
        generate = types.SimpleNamespace(generation_stream="import-thread-stream")
        server = types.SimpleNamespace(generation_stream="import-thread-stream")
        calls: list[str] = []

        class ResponseGenerator:
            def _generate(self) -> str:
                calls.append("generate")
                return "generated"

        server.ResponseGenerator = ResponseGenerator
        patch_mlx_lm_generation_worker(mlx, generate, server)

        self.assertEqual("generated", ResponseGenerator()._generate())
        self.assertEqual(["generate"], calls)
        self.assertEqual("thread-local:gpu", generate.generation_stream)
        self.assertEqual("thread-local:gpu", server.generation_stream)

    def test_dedicated_command_uses_adapter_qwen35_and_never_shared_8765(self) -> None:
        """FR-5: Harbor owns an isolated Qwen3.5 server at :8766."""
        command = server_command("Qwen/Qwen3.5-0.8B")

        self.assertEqual(
            [
                sys.executable,
                "-m",
                "omlx_research.harbor_mlx_server",
                "--model",
                "Qwen/Qwen3.5-0.8B",
                "--host",
                "0.0.0.0",
                "--port",
                "8766",
            ],
            command,
        )
        with self.assertRaises(HarborMlxServerError):
            server_command("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        with self.assertRaises(HarborMlxServerError):
            server_command("Qwen/Qwen3.5-0.8B", port=8765)


if __name__ == "__main__":
    unittest.main()

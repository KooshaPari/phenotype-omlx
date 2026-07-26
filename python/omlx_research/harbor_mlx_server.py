"""Dedicated Qwen3.5 MLX server adapter for Harbor NIAH runs.

The standard ``mlx_lm.server`` imports a generation stream on its import
thread, then runs generation on a dedicated worker.  Recent MLX releases make
GPU streams thread-local, so the worker must own and bind its stream before it
evaluates Qwen3.5 arrays.  This adapter patches only that worker and is
intended only for the Harbor-owned ``:8766`` endpoint.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any


DEDICATED_PORT = 8766
SHARED_PORT = 8765


class HarborMlxServerError(RuntimeError):
    """Invalid dedicated Harbor MLX server configuration."""


def _require_qwen35(model: str) -> None:
    if "qwen3.5" not in model.lower():
        raise HarborMlxServerError(
            "Harbor MLX server requires a Qwen3.5 model; Qwen2.5 is quarantined"
        )


def _require_dedicated_port(port: int) -> None:
    if port == SHARED_PORT:
        raise HarborMlxServerError("refusing shared :8765; Harbor owns dedicated :8766 only")
    if port != DEDICATED_PORT:
        raise HarborMlxServerError(f"Harbor MLX server must use dedicated :{DEDICATED_PORT}")


def server_command(model: str, *, host: str = "0.0.0.0", port: int = DEDICATED_PORT) -> list[str]:
    """Build, but do not execute, the dedicated thread-safe server command."""
    _require_qwen35(model)
    _require_dedicated_port(port)
    return [
        sys.executable,
        "-m",
        "omlx_research.harbor_mlx_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]


def install_worker_generation_stream(mlx: Any, generate_module: Any, server_module: Any) -> Any:
    """Create and bind an MLX stream in the generation worker's own thread."""
    factory = getattr(mlx, "new_thread_local_stream", None)
    if factory is None:
        raise HarborMlxServerError(
            "installed MLX lacks new_thread_local_stream; upgrade MLX before Harbor NIAH"
        )
    stream = factory(mlx.default_device())
    generate_module.generation_stream = stream
    # ``mlx_lm.server`` imports this name directly, so rebind both references.
    server_module.generation_stream = stream
    return stream


def patch_mlx_lm_generation_worker(mlx: Any, generate_module: Any, server_module: Any) -> None:
    """Patch only ``ResponseGenerator._generate`` to bind its local MLX stream."""
    response_generator = server_module.ResponseGenerator
    original = response_generator._generate
    if getattr(original, "_omlx_harbor_thread_stream", False):
        return

    def worker_with_thread_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
        install_worker_generation_stream(mlx, generate_module, server_module)
        return original(self, *args, **kwargs)

    worker_with_thread_stream._omlx_harbor_thread_stream = True  # type: ignore[attr-defined]
    response_generator._generate = worker_with_thread_stream


def run_server(model: str, *, host: str = "0.0.0.0", port: int = DEDICATED_PORT) -> None:
    """Install the worker-only patch then delegate to the installed MLX server."""
    _require_qwen35(model)
    _require_dedicated_port(port)
    try:
        import mlx.core as mx
        import mlx_lm.generate as generate_module
        import mlx_lm.server as server_module
    except ImportError as exc:
        raise HarborMlxServerError("mlx and mlx-lm are required for the Harbor MLX server") from exc

    patch_mlx_lm_generation_worker(mx, generate_module, server_module)
    argv = sys.argv
    sys.argv = ["mlx_lm.server", "--model", model, "--host", host, "--port", str(port)]
    try:
        server_module.main()
    finally:
        sys.argv = argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEDICATED_PORT)
    args = parser.parse_args(argv)
    try:
        run_server(args.model, host=args.host, port=args.port)
    except HarborMlxServerError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

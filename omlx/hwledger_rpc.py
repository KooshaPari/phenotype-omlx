#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
JSON-RPC 2.0 server over stdin/stdout for hwLedger inference integration.

Methods:
  - generate: {"prompt": str, "model": str, "max_tokens": int, "temperature": float, "stream": bool, "request_id": str}
  - cancel: {"request_id": str}
  - load_model: {"model": str, "max_kv_size": int}
  - unload_model: {"model": str}
  - memory_report: {} -> {"total_unified_mb": n, "used_by_mlx_mb": m, "kv_cache_mb": k, "loaded_models": [...]}
  - health: {} -> {"status":"ok","uptime_s":n,"mlx_version":"..."}

Protocol:
  - Line-delimited JSON (each message is a single line terminated with \n).
  - Requests: {"jsonrpc":"2.0","method":"...","params":{...},"id":n}
  - Responses: {"jsonrpc":"2.0","result":{...},"id":n} or {"jsonrpc":"2.0","error":{"code":-32000,"message":"...","data":{"traceback":"..."}},"id":n}
  - Notifications (for streaming): {"jsonrpc":"2.0","method":"token","params":{"request_id":"...","text":"..."}}
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
from uuid import UUID

import psutil

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


@dataclass
class GenerateRequest:
    prompt: str
    model: str
    max_tokens: int
    temperature: float
    stream: bool
    request_id: str


@dataclass
class TokenEvent:
    request_id: str
    text: str
    is_final: bool = False
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    stopped_reason: Optional[str] = None


class HwLedgerRpcServer:
    """JSON-RPC 2.0 server for hwLedger inference."""

    def __init__(self, engine_pool=None):
        """Initialize with optional engine pool (injected for testing)."""
        self.engine_pool = engine_pool
        self.start_time = time.time()
        self.next_request_id = 0
        self.running_generations: Dict[str, asyncio.Task] = {}
        self.generation_lock = threading.Lock()
        self.pending_tokens: Dict[str, list] = {}
        self.process = psutil.Process()

    async def generate(self, req: GenerateRequest) -> None:
        """Spawn an async generation task and stream tokens via JSON-RPC notifications."""
        request_id = req.request_id

        if self.engine_pool is None:
            await self._send_error(None, -32000, "Engine pool not initialized")
            return

        async def _generate_task():
            try:
                prompt_tokens = 0
                completion_tokens = 0
                stopped_reason = "eos"

                if self.engine_pool is not None:
                    # In a real deployment, this calls the actual mlx-lm engine.
                    # For now, we stub it out to accept any model and yield placeholder tokens.
                    for i in range(min(req.max_tokens, 10)):
                        # Simulated token: in production, this comes from the engine pool.
                        token_text = f"[token-{i}]"
                        completion_tokens += 1
                        await self._send_token_notification(request_id, token_text)
                        await asyncio.sleep(0.01)  # Simulate generation latency

                # Send final result
                await self._send_result(
                    request_id,
                    {
                        "request_id": request_id,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "stopped_reason": stopped_reason,
                    },
                )
            except Exception as e:
                logger.exception(f"Error in generate task for {request_id}: {e}")
                await self._send_error(request_id, -32000, str(e), traceback.format_exc())
            finally:
                with self.generation_lock:
                    self.running_generations.pop(request_id, None)
                    self.pending_tokens.pop(request_id, None)

        task = asyncio.create_task(_generate_task())
        with self.generation_lock:
            self.running_generations[request_id] = task
            self.pending_tokens[request_id] = []

    async def cancel(self, request_id: str) -> None:
        """Cancel a running generation by request_id."""
        with self.generation_lock:
            task = self.running_generations.pop(request_id, None)
            self.pending_tokens.pop(request_id, None)

        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def load_model(
        self, model: str, max_kv_size: int
    ) -> Dict[str, Any]:
        """Load a model into the engine pool."""
        if self.engine_pool is None:
            return {"loaded": False, "error": "Engine pool not initialized"}

        # Stub: in production, this calls engine_pool.load_model(model, max_kv_size).
        # For now, we pretend any model loads successfully.
        return {
            "loaded": True,
            "model": model,
            "context_length": 8192,
        }

    async def unload_model(self, model: str) -> Dict[str, Any]:
        """Unload a model from the engine pool."""
        if self.engine_pool is None:
            return {"unloaded": False, "error": "Engine pool not initialized"}

        # Stub: in production, this calls engine_pool.unload_model(model).
        return {"unloaded": True}

    async def memory_report(self) -> Dict[str, Any]:
        """Report unified memory usage breakdown."""
        try:
            mem_info = self.process.memory_info()
            total_mb = mem_info.rss / 1024 / 1024
            # Simplified: assume 30% is MLX, rest is overhead
            used_by_mlx_mb = total_mb * 0.3
            kv_cache_mb = total_mb * 0.5

            return {
                "total_unified_mb": round(total_mb, 2),
                "used_by_mlx_mb": round(used_by_mlx_mb, 2),
                "kv_cache_mb": round(kv_cache_mb, 2),
                "loaded_models": [],  # Would be populated from engine_pool in production
            }
        except Exception as e:
            logger.exception(f"Error in memory_report: {e}")
            return {
                "total_unified_mb": 0,
                "used_by_mlx_mb": 0,
                "kv_cache_mb": 0,
                "loaded_models": [],
                "error": str(e),
            }

    async def health(self) -> Dict[str, Any]:
        """Report server health."""
        uptime = time.time() - self.start_time
        return {
            "status": "ok",
            "uptime_s": round(uptime, 2),
            "mlx_version": "0.3.6",  # Placeholder; would be fetched from mlx-lm in production
        }

    async def handle_request(self, line: str) -> None:
        """Parse and dispatch a single JSON-RPC request line."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return

        method = msg.get("method")
        params = msg.get("params", {})
        request_id = msg.get("id")

        try:
            if method == "generate":
                gen_req = GenerateRequest(
                    prompt=params.get("prompt", ""),
                    model=params.get("model", ""),
                    max_tokens=params.get("max_tokens", 100),
                    temperature=params.get("temperature", 0.7),
                    stream=params.get("stream", True),
                    request_id=params.get("request_id", str(request_id)),
                )
                await self.generate(gen_req)

            elif method == "cancel":
                await self.cancel(params.get("request_id", ""))
                await self._send_result(request_id, {"cancelled": True})

            elif method == "load_model":
                result = await self.load_model(
                    model=params.get("model", ""),
                    max_kv_size=params.get("max_kv_size", 0),
                )
                await self._send_result(request_id, result)

            elif method == "unload_model":
                result = await self.unload_model(model=params.get("model", ""))
                await self._send_result(request_id, result)

            elif method == "memory_report":
                result = await self.memory_report()
                await self._send_result(request_id, result)

            elif method == "health":
                result = await self.health()
                await self._send_result(request_id, result)

            else:
                await self._send_error(request_id, -32601, f"Method not found: {method}")

        except Exception as e:
            logger.exception(f"Error handling {method}: {e}")
            await self._send_error(request_id, -32000, str(e), traceback.format_exc())

    async def _send_result(self, request_id: Any, result: Any) -> None:
        """Send a successful JSON-RPC result."""
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id,
        }
        print(json.dumps(response), file=sys.stdout, flush=True)

    async def _send_error(
        self, request_id: Any, code: int, message: str, data: Optional[str] = None
    ) -> None:
        """Send a JSON-RPC error response."""
        error_obj = {"code": code, "message": message}
        if data:
            error_obj["data"] = {"traceback": data}

        response = {
            "jsonrpc": "2.0",
            "error": error_obj,
            "id": request_id,
        }
        print(json.dumps(response), file=sys.stdout, flush=True)

    async def _send_token_notification(self, request_id: str, text: str) -> None:
        """Send a token notification (not awaiting an ID)."""
        notification = {
            "jsonrpc": "2.0",
            "method": "token",
            "params": {
                "request_id": request_id,
                "text": text,
            },
        }
        print(json.dumps(notification), file=sys.stdout, flush=True)

    async def run_stdin_loop(self) -> None:
        """Main event loop: read JSON-RPC requests from stdin, dispatch them."""
        loop = asyncio.get_running_loop()

        def read_stdin():
            """Blocking read from stdin; yields lines."""
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    line = line.rstrip("\n\r")
                    if line:
                        yield line
                except Exception as e:
                    logger.error(f"Error reading stdin: {e}")
                    break

        # Run stdin reading in a thread pool to avoid blocking the event loop.
        def stdin_reader_thread():
            for line in read_stdin():
                asyncio.run_coroutine_threadsafe(self.handle_request(line), loop)

        reader_thread = threading.Thread(target=stdin_reader_thread, daemon=True)
        reader_thread.start()

        # Keep the event loop alive.
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            sys.exit(0)


def main():
    """Entry point for hwLedger RPC server."""
    server = HwLedgerRpcServer()
    try:
        asyncio.run(server.run_stdin_loop())
    except KeyboardInterrupt:
        logger.info("RPC server shutdown")
        sys.exit(0)


if __name__ == "__main__":
    main()

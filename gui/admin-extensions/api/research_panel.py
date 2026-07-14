"""Research-panel API routes — backend status, agent dispatch, TurboQuant+ config.

This module exposes a Flask Blueprint (and an equivalent FastAPI APIRouter) that
the OMLX admin dashboard uses to query the state of every registered backend,
list and run research agents, and read/write TurboQuant+ configuration.

All heavy lifting is delegated to the ``omlx_research`` package so that the
same logic is shared between the CLI, the GUI, and the web admin panel.
"""

from __future__ import annotations

import json
import os
import time as _time
from dataclasses import asdict
from typing import Any

# ---------------------------------------------------------------------------
# Backend imports — every adapter the CLI / GUI uses is available here.
# ---------------------------------------------------------------------------
from omlx_research.backends import (
    BackendBase,
    BackendCapabilities,
    GenerateRequest,
    GenerateResponse,
    LlamaCppBackend,
    MetalKernelBackend,
    MlxBackend,
    SglangBackend,
    TensorrtBackend,
    VllmBackend,
)

from omlx_research.agents import (
    ConcurrentScheduler,
    JetSpecRunner,
    LatentMasRunner,
    SsdRunner,
    TidarRunner,
    Strategy,
    jetspec_draft_tree,
    latentmas_fanout,
    tidar_ar_diffusion_loop,
)

# ---------------------------------------------------------------------------
# Package-level registry of all known backends and agents.
# ---------------------------------------------------------------------------
_BACKENDS: list[BackendBase] = [
    MlxBackend(),
    MetalKernelBackend(),
    VllmBackend(),
    SglangBackend(),
    TensorrtBackend(),
    LlamaCppBackend(),
]

_AGENT_DESCRIPTIONS: list[dict[str, Any]] = [
    {
        "id": "latentmas",
        "label": "LatentMAS",
        "description": "Concurrent multi-agent fan-out (proposer / verifier / refiner / critic).",
    },
    {
        "id": "tidar",
        "label": "TiDAR",
        "description": "Think in Diffusion, Talk in AR — hybrid draft-verify loop.",
    },
    {
        "id": "ssd",
        "label": "SSD",
        "description": "Self-Speculative Decoding via n-gram prompt-lookup.",
    },
    {
        "id": "jetspec",
        "label": "JetSpec",
        "description": "Tree-attention speculative decoding with Medusa-style heads.",
    },
]

# ---------------------------------------------------------------------------
# Flask Blueprint
# ---------------------------------------------------------------------------
try:
    from flask import Blueprint, Response, jsonify, request

    research_bp = Blueprint(
        "research_panel",
        __name__,
        url_prefix="/api/research",
        template_folder="../templates",
        static_folder="../static",
        static_url_path="/static/admin-extensions",
    )
    _HAS_FLASK = True
except ImportError:  # pragma: no cover
    research_bp = None  # type: ignore[assignment]
    _HAS_FLASK = False

# ---------------------------------------------------------------------------
# FastAPI Router (fallback for ASGI-mode admin apps)
# ---------------------------------------------------------------------------
try:
    from fastapi import APIRouter

    research_router = APIRouter(prefix="/api/research")
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    research_router = None  # type: ignore[assignment]
    _HAS_FASTAPI = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend_status(b: BackendBase) -> dict[str, Any]:
    """Return a snapshot dict for a single backend adapter."""
    caps: BackendCapabilities = b.capabilities
    available = b.is_available()
    return {
        "id": caps.name,
        "name": caps.name,
        "primary": caps.primary,
        "available": available,
        "cuda": caps.cuda,
        "metal": caps.metal,
        "supports_batching": caps.supports_batching,
        "supports_streaming": caps.supports_streaming,
        "supports_turboquant": caps.supports_turboquant,
        "supports_spec_decode": caps.supports_spec_decode,
    }


def _all_backend_statuses() -> list[dict[str, Any]]:
    return [_backend_status(b) for b in _BACKENDS]


def _turboquant_config() -> dict[str, Any]:
    """Read the current TurboQuant+ configuration from environment / defaults."""
    return {
        "enabled": os.environ.get("TURBOQUANT_ENABLED", "0") == "1",
        "kv_cache_bits": int(os.environ.get("TURBOQUANT_KV_CACHE_BITS", "4")),
        "weight_bits": int(os.environ.get("TURBOQUANT_WEIGHT_BITS", "4")),
        "block_size": int(os.environ.get("TURBOQUANT_BLOCK_SIZE", "64")),
        "rotation_enabled": os.environ.get("TURBOQUANT_ROTATION", "1") == "1",
        "outlier_channel_threshold": float(
            os.environ.get("TURBOQUANT_OUTLIER_THRESHOLD", "2.0")
        ),
        "codebook_size": int(os.environ.get("TURBOQUANT_CODEBOOK_SIZE", "65536")),
    }


def _turboquant_apply_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply a (partial) TurboQuant+ config payload to the environment.

    For a production admin app this would persist to a config file or DB;
    in this reference implementation we round-trip through environment variables.
    """
    mapping = {
        "enabled": "TURBOQUANT_ENABLED",
        "kv_cache_bits": "TURBOQUANT_KV_CACHE_BITS",
        "weight_bits": "TURBOQUANT_WEIGHT_BITS",
        "block_size": "TURBOQUANT_BLOCK_SIZE",
        "rotation_enabled": "TURBOQUANT_ROTATION",
        "outlier_channel_threshold": "TURBOQUANT_OUTLIER_THRESHOLD",
        "codebook_size": "TURBOQUANT_CODEBOOK_SIZE",
    }
    for json_key, env_key in mapping.items():
        if json_key in payload:
            os.environ[env_key] = str(payload[json_key])
    return _turboquant_config()


def _try_import_turboquant() -> dict[str, Any]:
    """Return TurboQuant+ import diagnostics (mirrors ``cmd_doctor``)."""
    result: dict[str, Any] = {"installed": False, "version": None, "path": None}
    try:
        import turboquant_plus  # type: ignore[import-untyped,unused-ignore]

        result["installed"] = True
        result["version"] = getattr(turboquant_plus, "__version__", "unknown")
        result["path"] = getattr(turboquant_plus, "__file__", None)
    except ImportError:
        try:
            candidate = "/Users/kooshapari/CodeProjects/Phenotype/repos/turboquant_plus"
            if os.path.isdir(candidate):
                result["installed"] = False
                result["path"] = candidate
                result["version"] = "present (not activated)"
        except Exception:
            pass
    try:
        from mlx.nn.layers.turbo_kv_cache import TurboKVCache  # type: ignore[import-untyped,unused-ignore]

        result["turbo_kv_cache"] = True
    except ImportError:
        result["turbo_kv_cache"] = False
    return result


# ---------------------------------------------------------------------------
# Flask route registrations
# ---------------------------------------------------------------------------


def _register_flask_routes() -> None:
    if not _HAS_FLASK:
        return

    @research_bp.route("/status", methods=["GET"])
    def status() -> Response:
        """Return status of all backends + TurboQuant+ availability."""
        return jsonify(
            {
                "backends": _all_backend_statuses(),
                "turboquant": _try_import_turboquant(),
                "turboquant_config": _turboquant_config(),
                "timestamp": _time.time(),
            }
        )

    @research_bp.route("/agents/list", methods=["GET"])
    def agents_list() -> Response:
        """Return the list of available research agents."""
        return jsonify({"agents": _AGENT_DESCRIPTIONS, "count": len(_AGENT_DESCRIPTIONS)})

    @research_bp.route("/agents/run", methods=["POST"])
    def agents_run() -> Response:
        """Run a named agent with a JSON body containing ``agent`` and ``prompt``.

        Request body::

            {"agent": "latentmas", "prompt": "...", "params": {}}

        Response::

            {"ok": true, "agent": "...", "output": "...", "elapsed_ms": 123}
        """
        data: dict[str, Any] = request.get_json(silent=True) or {}
        agent_id: str = data.get("agent", "")
        prompt: str = data.get("prompt", "")
        params: dict[str, Any] = data.get("params", {})

        if not agent_id:
            return jsonify({"ok": False, "error": "missing 'agent' field"}), 400
        if not prompt:
            return jsonify({"ok": False, "error": "missing 'prompt' field"}), 400

        t0 = _time.time()
        try:
            output = _dispatch_agent(agent_id, prompt, params)
            elapsed = int((_time.time() - t0) * 1000)
            return jsonify(
                {
                    "ok": True,
                    "agent": agent_id,
                    "output": str(output),
                    "elapsed_ms": elapsed,
                }
            )
        except Exception as exc:
            elapsed = int((_time.time() - t0) * 1000)
            return jsonify(
                {
                    "ok": False,
                    "agent": agent_id,
                    "error": str(exc),
                    "elapsed_ms": elapsed,
                }
            ), 500

    @research_bp.route("/turboquant/config", methods=["GET", "POST"])
    def turboquant_config() -> Response:
        """GET returns current TurboQuant+ config; POST applies a (partial) config."""
        if request.method == "POST":
            payload: dict[str, Any] = request.get_json(silent=True) or {}
            updated = _turboquant_apply_config(payload)
            return jsonify({"ok": True, "config": updated})
        return jsonify({"ok": True, "config": _turboquant_config()})

    @research_bp.route("/turboquant/diagnostics", methods=["GET"])
    def turboquant_diagnostics() -> Response:
        """Return TurboQuant+ import / runtime diagnostics."""
        return jsonify(_try_import_turboquant())


_register_flask_routes()

# ---------------------------------------------------------------------------
# FastAPI route registrations
# ---------------------------------------------------------------------------
# We register equivalent routes on the FastAPI router so the same endpoints
# work under ASGI mode.  The decorators use ``@research_router.get`` / ``.post``
# style; function bodies are identical but reference FastAPI's ``Request``.

try:
    from fastapi.responses import JSONResponse

    if _HAS_FASTAPI and research_router is not None:

        @research_router.get("/status")
        async def fastapi_status() -> JSONResponse:
            from fastapi import Request as FastRequest

            return JSONResponse(
                content={
                    "backends": _all_backend_statuses(),
                    "turboquant": _try_import_turboquant(),
                    "turboquant_config": _turboquant_config(),
                    "timestamp": _time.time(),
                }
            )

        @research_router.get("/agents/list")
        async def fastapi_agents_list() -> JSONResponse:
            return JSONResponse(
                content={"agents": _AGENT_DESCRIPTIONS, "count": len(_AGENT_DESCRIPTIONS)}
            )

        @research_router.post("/agents/run")
        async def fastapi_agents_run(request: Any) -> JSONResponse:
            # request is a FastAPI Request — read JSON body.
            from fastapi import Request as FastRequest

            body = await request.json() if hasattr(request, "json") else {}
            agent_id: str = body.get("agent", "")
            prompt: str = body.get("prompt", "")
            params: dict[str, Any] = body.get("params", {})

            if not agent_id:
                return JSONResponse(content={"ok": False, "error": "missing 'agent' field"}, status_code=400)
            if not prompt:
                return JSONResponse(content={"ok": False, "error": "missing 'prompt' field"}, status_code=400)

            t0 = _time.time()
            try:
                output = _dispatch_agent(agent_id, prompt, params)
                elapsed = int((_time.time() - t0) * 1000)
                return JSONResponse(
                    content={
                        "ok": True,
                        "agent": agent_id,
                        "output": str(output),
                        "elapsed_ms": elapsed,
                    }
                )
            except Exception as exc:
                elapsed = int((_time.time() - t0) * 1000)
                return JSONResponse(
                    content={
                        "ok": False,
                        "agent": agent_id,
                        "error": str(exc),
                        "elapsed_ms": elapsed,
                    },
                    status_code=500,
                )

        @research_router.get("/turboquant/config")
        async def fastapi_tq_config_get() -> JSONResponse:
            return JSONResponse(content={"ok": True, "config": _turboquant_config()})

        @research_router.post("/turboquant/config")
        async def fastapi_tq_config_post(request: Any) -> JSONResponse:
            body = await request.json() if hasattr(request, "json") else {}
            updated = _turboquant_apply_config(body)
            return JSONResponse(content={"ok": True, "config": updated})

        @research_router.get("/turboquant/diagnostics")
        async def fastapi_tq_diagnostics() -> JSONResponse:
            return JSONResponse(content=_try_import_turboquant())

except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Agent dispatch — shared logic for both Flask and FastAPI handlers
# ---------------------------------------------------------------------------


def _dispatch_agent(agent_id: str, prompt: str, params: dict[str, Any]) -> Any:
    """Route a prompt to the appropriate research agent.

    Parameters
    ----------
    agent_id : str
        One of ``"latentmas"``, ``"tidar"``, ``"ssd"``, ``"jetspec"``.
    prompt : str
        Input text for the agent.
    params : dict
        Agent-specific keyword arguments (e.g. ``n_agents`` for LatentMAS).

    Returns
    -------
    Any
        Agent-specific result (usually a string or summary dict).
    """
    import asyncio

    agent_id = agent_id.lower().strip()

    if agent_id == "latentmas":
        n_agents: int = int(params.get("n_agents", 4))

        async def _run_latentmas() -> list[str]:
            async def _stub_agent(_p: str, _s: dict, idx: int) -> str:
                await asyncio.sleep(0.01)  # simulate work
                return f"[LatentMAS agent-{idx} processed: {_p[:64]}]"

            fns = [
                lambda p, s, i=i: _stub_agent(p, s, i) for i in range(n_agents)
            ]
            return await latentmas_fanout(fns, prompt, {})

        return asyncio.run(_run_latentmas())

    if agent_id == "tidar":
        draft_len: int = int(params.get("draft_len", 4))
        steps: int = int(params.get("steps", 8))

        async def _run_tidar() -> list[int]:
            # Stub callables — real integration wires actual model logits.
            async def _stub_base_lm(tokens: list[int]) -> list[float]:
                return [float(len(tokens) + i) for i in range(32)]

            async def _stub_drafter(tokens: list[int]) -> list[int]:
                return [min(t + 1, 31) for t in tokens[-draft_len:]]

            async def _stub_verifier(
                tokens: list[int], draft: list[int]
            ) -> list[int]:
                return [1 if t > 0 else 0 for t in draft]

            runner = TidarRunner(
                base_lm=_stub_base_lm,
                drafter=_stub_drafter,
                verifier=_stub_verifier,
                draft_len=draft_len,
                steps=steps,
            )
            # Convert prompt string to toy token ids.
            token_ids = [ord(c) % 32 for c in prompt[:64]] or [1]
            return await runner(token_ids)

        return asyncio.run(_run_tidar())

    if agent_id == "ssd":
        gamma: int = int(params.get("gamma", 5))

        async def _run_ssd() -> list[int]:
            # Stub target that returns uniform logits.
            async def _stub_target(tokens: list[int]) -> list[float]:
                return [1.0] * 32

            runner = SsdRunner(target=_stub_target, gamma=gamma)
            prefix = [ord(c) % 32 for c in prompt[:32]] or [1]
            return await runner.step(prefix)

        return asyncio.run(_run_ssd())

    if agent_id == "jetspec":
        width: int = int(params.get("width", 4))
        depth: int = int(params.get("depth", 2))

        async def _run_jetspec() -> list[int]:
            # Stub: generate a toy draft tree and verify.
            def _head_fn(tok_idx: int) -> int:
                return tok_idx % 32

            prefix = [ord(c) % 32 for c in prompt[:16]] or [1]
            tree = jetspec_draft_tree(width, depth, _head_fn, prefix)

            async def _stub_target(tokens: list[int]) -> list[float]:
                return [1.0] * 32

            runner = JetSpecRunner(
                target=_stub_target, draft_tree=tree, width=width, depth=depth
            )
            return await runner.step(prefix)

        return asyncio.run(_run_jetspec())

    raise ValueError(f"Unknown agent: {agent_id!r}. Expected one of: latentmas, tidar, ssd, jetspec.")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

__all__ = [
    "research_bp",
    "research_router",
    "_BACKENDS",
    "_AGENT_DESCRIPTIONS",
    "_backend_status",
    "_all_backend_statuses",
    "_turboquant_config",
    "_turboquant_apply_config",
    "_try_import_turboquant",
    "_dispatch_agent",
]

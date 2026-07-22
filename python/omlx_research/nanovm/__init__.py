"""NanoVM plugin system — file-based plugin discovery with proper domain decomposition.

Domain decomposition:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  BACKEND  (kind = "backend")   model-serving runtime, loads weights, runs │
  │                               forward pass.  Native to one accelerator.   │
  │     • mlx-metal   — Apple MLX + Metal GPU (Apple Silicon)                │
  │     • sglang      — SGLang (primary, cross-platform)                     │
  │     • vllm        — vLLM (NVIDIA, ROCm experimental)                     │
  │     • tensorrt    — NVIDIA TensorRT-LLM                                  │
  │     • llamacpp    — llama.cpp (CPU + Metal + CUDA, universal fallback)   │
  │                                                                          │
  │  STRATEGY (kind = "strategy")  flippable multi-agent / decoding wrapper. │
  │                                Wraps ONE OR MORE backends, declares      │
  │                                pipeline PHASES that need backends.       │
  │     • latentmas   — multi-agent latent reasoning (debate→propose→vote)   │
  │     • tidar       — hybrid AR + diffusion (draft→diffuse→verify)        │
  │     • ssd         — self-speculative decoding (draft→verify)             │
  │     • jetspec     — tree-attention spec-decode (tree_draft→tree_verify)  │
  │                                                                          │
  │  PIPELINE  =  sequence of (phase, backend) pairs produced by a strategy │
  │               given a backend pool. The orchestrator executes the        │
  │               pipeline using native backends per phase (no PyTorch      │
  │               fallback unless requested).                                │
  │                                                                          │
  │  COMPOSITION  HybridOrchestrator.run_all(strategies=[...], pool=[...])  │
  │               runs multiple strategies concurrently, each with their     │
  │               own pipeline built from the shared backend pool.           │
  └──────────────────────────────────────────────────────────────────────────┘

Why file-based rather than entry-points?
  - Zero setup: drop a directory, get a plugin
  - Cross-platform: works on macOS, Linux, Windows identically
  - Hot-reloadable: re-discover at any time
  - Discoverable: `omlx-research nanovm list` shows everything
"""
from __future__ import annotations

import importlib
import logging
import os
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Domain kind ─────────────────────────────────────────────────────────────

class PluginKind(str, Enum):
    """Whether this plugin is a BACKEND (model-serving) or STRATEGY (multi-agent wrapper)."""
    BACKEND = "backend"     # loads weights, runs forward pass
    STRATEGY = "strategy"   # wraps backends, declares phases


class BackendKind(str, Enum):
    """Which model-serving runtime / accelerator the plugin wraps."""
    MLX_METAL = "mlx_metal"      # Apple MLX (Metal GPU)
    SGLANG = "sglang"            # SGLang (primary cross-platform)
    VLLM = "vllm"                # vLLM (NVIDIA primary, ROCm experimental)
    TENSORRT = "tensorrt"        # NVIDIA TensorRT-LLM
    LLAMACPP = "llamacpp"        # llama.cpp (CPU + Metal + CUDA)
    TURBOQUANT = "turboquant"    # TurboQuant+ KV compression (MLX)


class StrategyKind(str, Enum):
    """Which multi-agent / decoding strategy the plugin implements."""
    LATENTMAS = "latentmas"      # Multi-agent latent reasoning
    TIDAR = "tidar"              # Hybrid AR + diffusion
    SSD = "ssd"                  # Self-speculative decoding
    JETSPEC = "jetspec"          # Tree-attention speculative decoding


# ── Plugin spec ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PluginSpec:
    """Static description of a NanoVM plugin loaded from plugin.toml.

    Common fields:
      name, kind, module, description, version, requires, path

    Backend-specific fields (when kind=BACKEND):
      runtime   — the model-serving runtime (mlx | sglang | vllm | tensorrt | llamacpp)
      priority  — 1-10, higher = preferred when multiple backends match

    Strategy-specific fields (when kind=STRATEGY):
      phases             — list of phase names the strategy executes (in order)
      compatible_backends — list of backend kinds this strategy can wrap
      parallel           — True if the strategy's phases can run in parallel
      default_pool       — recommended backend kinds per phase (length == len(phases))
    """
    name: str
    kind: PluginKind
    path: Path
    module: str
    runtime: str = "unknown"
    priority: int = 5
    description: str = ""
    version: str = "0.0.0"
    requires: list[str] = field(default_factory=list)

    # Strategy-only (ignored for backends):
    phases: list[str] = field(default_factory=list)
    compatible_backends: list[str] = field(default_factory=list)
    parallel: bool = False
    default_pool: list[str] = field(default_factory=list)

    def can_load(self) -> tuple[bool, str]:
        """Check runtime requirements. Returns (ok, reason)."""
        def _vtuple(s: str) -> tuple[int, ...]:
            """Parse first 3 numeric components from a PEP 440 version string.

            Handles dev versions like '0.31.2.dev20260707+6305521' by taking
            only the leading numeric dotted components and ignoring suffixes.
            """
            out: list[int] = []
            for piece in s.split("."):
                digits = ""
                for ch in piece:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    out.append(int(digits))
                else:
                    break
                if len(out) >= 3:
                    break
            return tuple(out)

        # Map pip distribution names → Python module names. PyPI uses hyphens;
        # the import system uses underscores. Without this map, "mlx-lm"
        # (a valid pip package) is treated as missing because
        # importlib.import_module("mlx-lm") raises ImportError.
        _DIST_TO_MODULE = {
            "mlx-lm": "mlx_lm",
            "mlx-lm-nightly": "mlx_lm",
            "transformers": "transformers",
            "vllm": "vllm",
            "tensorrt-llm": "tensorrt_llm",
            "sglang": "sglang",
            "llama-cpp-python": "llama_cpp",
            "huggingface-hub": "huggingface_hub",
            "sentence-transformers": "sentence_transformers",
        }
        for req in self.requires:
            tag, _, version = req.partition(">=")
            tag = tag.strip()
            module_name = _DIST_TO_MODULE.get(tag, tag.replace("-", "_"))
            try:
                mod = importlib.import_module(module_name)
            except ImportError:
                # Fallback to original tag (underscored) before declaring missing
                try:
                    mod = importlib.import_module(tag.replace("-", "_"))
                except ImportError:
                    return False, f"missing dependency: {tag}"
            if version and hasattr(mod, "__version__"):
                have = _vtuple(mod.__version__)
                need = _vtuple(version)
                # Pad to same length with 0s
                n = max(len(have), len(need))
                have = have + (0,) * (n - len(have))
                need = need + (0,) * (n - len(need))
                if have < need:
                    return False, f"{tag} {version}+ required (have {mod.__version__})"
        return True, "ok"


class PluginRegistry:
    """Process-wide registry of all discovered NanoVM plugins."""

    def __init__(self):
        self._plugins: dict[str, PluginSpec] = {}

    def register(self, spec: PluginSpec) -> None:
        if spec.name in self._plugins:
            logger.debug("plugin %s already registered, overwriting", spec.name)
        self._plugins[spec.name] = spec

    def get(self, name: str) -> PluginSpec | None:
        return self._plugins.get(name)

    def all(self) -> list[PluginSpec]:
        return list(self._plugins.values())

    def backends(self) -> list[PluginSpec]:
        return [p for p in self._plugins.values() if p.kind == PluginKind.BACKEND]

    def strategies(self) -> list[PluginSpec]:
        return [p for p in self._plugins.values() if p.kind == PluginKind.STRATEGY]

    def backends_by_kind(self, kind: str) -> list[PluginSpec]:
        return [p for p in self.backends() if p.runtime == kind or p.name == kind]

    def by_kind(self, kind: PluginKind) -> list[PluginSpec]:
        return [p for p in self._plugins.values() if p.kind == kind]

    def clear(self) -> None:
        self._plugins.clear()

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins


_REGISTRY = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Return the process-wide plugin registry."""
    return _REGISTRY


# ── Discovery ───────────────────────────────────────────────────────────────

def discover_plugins(root: str | Path | None = None) -> PluginRegistry:
    """Walk `root` for `plugin.toml` files and register them all.

    Default root is the bundled `plugins/` dir next to this module.
    Set `OMLX_NANOVM_PLUGINS=/path/to/extra` to add more.
    """
    roots: list[Path] = []
    if root is not None:
        roots.append(Path(root))
    else:
        roots.append(Path(__file__).parent / "plugins")
    env_root = os.environ.get("OMLX_NANOVM_PLUGINS")
    if env_root:
        roots.append(Path(env_root))

    for r in roots:
        if not r.is_dir():
            continue
        for toml in r.glob("**/plugin.toml"):
            try:
                spec = _load_plugin(toml)
            except Exception as e:
                logger.warning("failed to load plugin %s: %s", toml, e)
                continue
            _REGISTRY.register(spec)
    return _REGISTRY


def _load_plugin(toml_path: Path) -> PluginSpec:
    data = tomllib.loads(toml_path.read_text())
    # Schema: [plugin] section holds all metadata.
    p = data.get("plugin", data)
    if "name" not in p:
        raise KeyError(
            f"{toml_path}: missing [plugin] section or 'name' field"
        )
    name = p["name"]
    kind = PluginKind(p["kind"])
    runtime = p.get("runtime", "unknown")
    priority = int(p.get("priority", 5))
    module = p["module"]
    description = p.get("description", "")
    version = p.get("version", "0.0.0")
    requires = list(p.get("requires", []))

    # Strategy-specific fields
    phases = list(p.get("phases", []))
    compatible_backends = list(p.get("compatible_backends", []))
    parallel = bool(p.get("parallel", False))
    default_pool = list(p.get("default_pool", []))

    return PluginSpec(
        name=name,
        kind=kind,
        runtime=runtime,
        priority=priority,
        path=toml_path.parent,
        module=module,
        description=description,
        version=version,
        requires=requires,
        phases=phases,
        compatible_backends=compatible_backends,
        parallel=parallel,
        default_pool=default_pool,
    )


# ── Convenience queries ────────────────────────────────────────────────────

def list_backends(include_unloadable: bool = False) -> list[dict[str, Any]]:
    """Return summary info for each registered BACKEND plugin."""
    out = []
    for spec in _REGISTRY.backends():
        ok, reason = spec.can_load()
        if not ok and not include_unloadable:
            continue
        out.append({
            "name": spec.name,
            "runtime": spec.runtime,
            "priority": spec.priority,
            "version": spec.version,
            "description": spec.description,
            "loadable": ok,
            "reason": reason,
        })
    return out


def list_strategies(include_unloadable: bool = False) -> list[dict[str, Any]]:
    """Return summary info for each registered STRATEGY plugin."""
    out = []
    for spec in _REGISTRY.strategies():
        ok, reason = spec.can_load()
        if not ok and not include_unloadable:
            continue
        out.append({
            "name": spec.name,
            "phases": spec.phases,
            "compatible_backends": spec.compatible_backends,
            "parallel": spec.parallel,
            "default_pool": spec.default_pool,
            "priority": spec.priority,
            "version": spec.version,
            "description": spec.description,
            "loadable": ok,
            "reason": reason,
        })
    return out


def list_available_backends(include_unloadable: bool = False) -> list[dict[str, Any]]:
    """Backward-compat alias for list_backends()."""
    return list_backends(include_unloadable=include_unloadable)


def list_available_strategies() -> list[str]:
    """Backward-compat: just names of loaded strategies."""
    return [s.name for s in _REGISTRY.strategies()]


__all__ = [
    "PluginKind",
    "BackendKind",
    "StrategyKind",
    "PluginSpec",
    "PluginRegistry",
    "discover_plugins",
    "get_registry",
    "list_backends",
    "list_strategies",
    "list_available_backends",
    "list_available_strategies",
]
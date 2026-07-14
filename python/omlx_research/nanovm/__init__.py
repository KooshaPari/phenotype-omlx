"""NanoVM plugin system — lightweight, file-based plugin discovery.

A NanoVM plugin is a directory containing a `plugin.toml` describing:
  - Backend name (e.g. "mlx-metal", "sglang")
  - Backend kind (what device it runs on)
  - Runtime tags (mlx, pytorch, metal, cuda, rocm, cpu)
  - Priority (1-10, higher = preferred)
  - Path to the Python module implementing the plugin

Discovery walks a root directory looking for `plugin.toml` files. Each
plugin is loaded into a `PluginRegistry`. The hybrid orchestrator queries
the registry to find candidate backends for a given strategy.

Why file-based rather than entry-points?
  - Zero setup: drop a directory, get a plugin
  - Cross-platform: works on macOS, Linux, Windows identically
  - Hot-reloadable: re-discover at any time
  - Discoverable: `omlx-research nanovm list` shows everything

Examples:
    >>> from omlx_research.nanovm import discover_plugins, list_available_backends
    >>> discover_plugins()
    >>> backends = list_available_backends()
    >>> for b in backends: print(b['name'], b['kind'])
"""
from __future__ import annotations

import importlib
import logging
import os
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BackendKind(str, Enum):
    """Which backend / inference engine the plugin wraps."""
    MLX_METAL = "mlx_metal"      # Apple MLX (Metal GPU)
    SGLANG = "sglang"            # SGLang (primary cross-platform)
    VLLM = "vllm"                # vLLM (NVIDIA primary, ROCm experimental)
    TENSORRT = "tensorrt"        # NVIDIA TensorRT-LLM
    LLAMACPP = "llamacpp"        # llama.cpp (CPU + Metal + CUDA)
    LATENTMAS = "latentmas"      # Multi-agent latent reasoning
    TIDAR = "tidar"              # Hybrid AR + diffusion
    SSD = "ssd"                  # Self-speculative decoding (CUDA-only ref)
    JETSPEC = "jetspec"          # Tree-attention speculative decoding
    TURBOQUANT = "turboquant"    # TurboQuant+ KV compression


@dataclass(frozen=True)
class PluginSpec:
    """Static description of a NanoVM plugin loaded from plugin.toml."""
    name: str                                    # unique id, e.g. "mlx-metal"
    kind: BackendKind                            # which inference backend
    runtime: str                                 # "mlx" | "pytorch" | "rust" | "cpp"
    priority: int                                # 1-10, higher = preferred
    path: Path                                   # directory containing plugin.toml
    module: str                                  # python module path to import
    description: str = ""
    version: str = "0.0.0"
    requires: list[str] = field(default_factory=list)   # runtime requirements

    def can_load(self) -> tuple[bool, str]:
        """Check runtime requirements. Returns (ok, reason)."""
        for req in self.requires:
            tag, _, version = req.partition(">=")
            tag = tag.strip()
            try:
                mod = importlib.import_module(tag)
                if version and hasattr(mod, "__version__"):
                    if tuple(map(int, mod.__version__.split(".")[:2])) < tuple(map(int, version.split(".")[:2])):
                        return False, f"{tag} {version}+ required (have {mod.__version__})"
            except ImportError:
                return False, f"missing dependency: {tag}"
        return True, "ok"


class PluginRegistry:
    """Singleton-style registry of all discovered NanoVM plugins."""

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

    def by_kind(self, kind: BackendKind) -> list[PluginSpec]:
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
    name = data["name"]
    kind = BackendKind(data["kind"])
    runtime = data["runtime"]
    priority = int(data.get("priority", 5))
    module = data["module"]
    description = data.get("description", "")
    version = data.get("version", "0.0.0")
    requires = list(data.get("requires", []))
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
    )


def list_available_backends(include_unloadable: bool = False) -> list[dict[str, Any]]:
    """Return summary info for each registered plugin."""
    out = []
    for spec in _REGISTRY.all():
        ok, reason = spec.can_load()
        if not ok and not include_unloadable:
            continue
        out.append({
            "name": spec.name,
            "kind": spec.kind.value,
            "runtime": spec.runtime,
            "priority": spec.priority,
            "version": spec.version,
            "description": spec.description,
            "loadable": ok,
            "reason": reason,
        })
    return out


def list_available_strategies() -> list[str]:
    """Names of the orchestration strategies exposed by hybrid.orchestrator."""
    return [
        "PARALLEL_VOTE",
        "SPECULATIVE_DRAFT",
        "TIER_FALLBACK",
        "ADAPTIVE_RACE",
    ]


__all__ = [
    "BackendKind",
    "PluginSpec",
    "PluginRegistry",
    "discover_plugins",
    "get_registry",
    "list_available_backends",
    "list_available_strategies",
]
"""SSOT smoke-model resolution for phenotype-omlx.

Reads ``config/smoke_models.json``. Enforces Qwen3.5 for acceptance paths.
Qwen2.5 is quarantined behind ``OMLX_ALLOW_LEGACY_QWEN25=1``.

Mature evals belong in **Portage/Harbor** + ``harbor-langsmith``, not ad-hoc
scripts. Prefer ``scripts/evals/run_via_harbor.sh``.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "smoke_models.json"

_LEGACY_ESCAPE = "OMLX_ALLOW_LEGACY_QWEN25"
_MODEL_ENV = "OMLX_READY_MODEL"


class SmokeModelError(RuntimeError):
    """Invalid or quarantined model selection."""


def repo_root() -> Path:
    return _REPO_ROOT


def config_path() -> Path:
    return _CONFIG_PATH


@lru_cache(maxsize=1)
def load_smoke_config() -> dict[str, Any]:
    path = _CONFIG_PATH
    if not path.is_file():
        raise SmokeModelError(f"missing smoke model SSOT: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SmokeModelError(f"invalid smoke_models.json at {path}")
    return data


def _legacy_allowed() -> bool:
    return os.environ.get(_LEGACY_ESCAPE, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def assert_qwen35(model: str, *, allow_legacy: bool | None = None) -> str:
    """Reject non-Qwen3.5 unless legacy escape is set."""
    if allow_legacy is None:
        allow_legacy = _legacy_allowed()
    lower = model.lower()
    is_qwen35 = "qwen3.5" in lower
    is_qwen25 = "qwen2.5" in lower
    is_bare_qwen3 = "qwen3" in lower and "qwen3.5" not in lower
    if is_qwen25 or is_bare_qwen3:
        if not allow_legacy:
            cfg = load_smoke_config()
            legacy = (cfg.get("legacy_quarantine") or {}).get("qwen25_mlx", "")
            defaults = cfg.get("defaults") or {}
            raise SmokeModelError(
                f"refuses non-Qwen3.5 model {model!r}; "
                f"default mlx_hf={defaults.get('mlx_hf')!r}. "
                f"Quarantined legacy={legacy!r} — set {_LEGACY_ESCAPE}=1 "
                f"only for local debug (not FR acceptance)."
            )
    if not is_qwen35 and not allow_legacy:
        raise SmokeModelError(
            f"requires Qwen3.5 in model id (got {model!r})"
        )
    return model


def default_model_for(role: str = "readiness") -> str:
    """Resolve model id for a named role from SSOT (+ optional env override)."""
    override = os.environ.get(_MODEL_ENV, "").strip()
    if override:
        return assert_qwen35(override)

    cfg = load_smoke_config()
    defaults = cfg.get("defaults") or {}
    roles = cfg.get("roles") or {}
    key = roles.get(role, "mlx_hf")
    if key not in defaults:
        raise SmokeModelError(
            f"role {role!r} maps to unknown defaults key {key!r} in {config_path()}"
        )
    model = str(defaults[key]).strip()
    return assert_qwen35(model)


def default_mlx_model() -> str:
    return default_model_for("readiness")


def portage_settings() -> dict[str, Any]:
    cfg = load_smoke_config()
    return dict(cfg.get("portage") or {})


def require_portage_root() -> Path:
    """PORTAGE_ROOT must be set — no hardcoded worktree paths."""
    raw = os.environ.get("PORTAGE_ROOT", "").strip()
    settings = portage_settings()
    if not raw:
        if settings.get("require_portage_root", True):
            raise SmokeModelError(
                "PORTAGE_ROOT is required (portage-TEMP / Harbor checkout). "
                "Example: export PORTAGE_ROOT="
                "$HOME/CodeProjects/Phenotype/repos/worktrees/portage/<topic>"
            )
        raise SmokeModelError("PORTAGE_ROOT empty")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise SmokeModelError(f"PORTAGE_ROOT is not a directory: {path}")
    return path


def cli_print_model(argv: list[str] | None = None) -> int:
    """``python -m omlx_research.smoke_models [role]`` → print model id."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    role = args[0] if args else "readiness"
    try:
        print(default_model_for(role))
        return 0
    except SmokeModelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli_print_model())

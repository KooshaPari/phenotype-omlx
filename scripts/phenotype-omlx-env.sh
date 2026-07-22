#!/usr/bin/env bash
# phenotype-omlx environment activation script.
#
# Adds the perf-core (Rust) Python path, repo-local packages, and the
# research reference repos (TurboQuant+, JetSpec, SSD, LatentMAS, TiDAR) to
# PYTHONPATH and PATH. Designed to be `source`d from `omlx-research` or any
# ad-hoc shell:
#
#   source scripts/phenotype-omlx-env.sh
#
# When the active Python is 3.11 (matching OMLX.app's bundled Python), the
# OMLX framework's site-packages is prepended to PYTHONPATH so the bundled
# MLX + TurboKVCache are used. On Python 3.12+ (system venv), the OMLX
# framework is *not* injected because its numpy/mlx are compiled for 3.11
# and would shadow the 3.12 venv's site-packages.

if [[ -n "${ZSH_VERSION:-}" ]]; then
    # In zsh, %x resolves the currently sourced file. Keep it inside eval so
    # Bash never parses zsh's prompt-style parameter expansion.
    PHENOTYPE_OMLX_ENV_SOURCE="$(eval 'printf "%s" "${(%):-%x}"')"
else
    PHENOTYPE_OMLX_ENV_SOURCE="${BASH_SOURCE[0]}"
fi
PHENOTYPE_OMLX_ENV_DIR="$(cd -- "$(dirname -- "${PHENOTYPE_OMLX_ENV_SOURCE}")" && pwd -P)"
PHENOTYPE_OMLX_DEFAULT_HOME="$(cd -- "${PHENOTYPE_OMLX_ENV_DIR}/.." && pwd -P)"
PHENOTYPE_OMLX_HOME="${PHENOTYPE_OMLX_HOME:-${PHENOTYPE_OMLX_DEFAULT_HOME}}"
REPOS_ROOT="${REPOS_ROOT:-$(cd -- "${PHENOTYPE_OMLX_HOME}/.." && pwd -P)}"
OMLX_APP="${OMLX_APP:-/Applications/oMLX.app}"
OMLX_FRAMEWORK_DIR="${OMLX_APP}/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages"

# Prefer a native Python 3.14+ interpreter for local MLX work. The historical
# TurboQuant venv may be absent after recovery, so expose a stable executable
# instead of sourcing a nonexistent environment. Call it explicitly as
# `"$PHENOTYPE_OMLX_PYTHON" scripts/e2e_real_model.py`.
if [[ -z "${PHENOTYPE_OMLX_PYTHON:-}" ]]; then
    if command -v python3.14 >/dev/null 2>&1; then
        PHENOTYPE_OMLX_PYTHON="$(command -v python3.14)"
    elif command -v python3.15 >/dev/null 2>&1; then
        PHENOTYPE_OMLX_PYTHON="$(command -v python3.15)"
    else
        PHENOTYPE_OMLX_PYTHON="$(command -v python3 2>/dev/null || true)"
    fi
fi
export PHENOTYPE_OMLX_PYTHON

# Local benchmark invocations must not silently resolve or download mutable
# artifacts. Set PHENOTYPE_OMLX_OFFLINE=0 explicitly when network use is
# intentional and reviewed.
if [[ "${PHENOTYPE_OMLX_OFFLINE:-1}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# Detect the active Python's major.minor.
ACTIVE_PY_MINOR=""
if [[ -n "${PHENOTYPE_OMLX_PYTHON}" && -x "${PHENOTYPE_OMLX_PYTHON}" ]]; then
    ACTIVE_PY_MINOR=$("${PHENOTYPE_OMLX_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")
fi

# 1) OMLX framework: only inject when the active Python matches (3.11).
if [[ -n "$ACTIVE_PY_MINOR" && "$ACTIVE_PY_MINOR" == "3.11" && -d "$OMLX_FRAMEWORK_DIR" ]]; then
    export OMLX_FRAMEWORK="$OMLX_FRAMEWORK_DIR"
    case ":${PYTHONPATH:-}:" in
        *":${OMLX_FRAMEWORK}::"*) ;;
        *) export PYTHONPATH="${OMLX_FRAMEWORK}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
fi
# Persistent TurboQuant+ wrapper (survives OMLX updates) — used when the
# OMLX framework is *not* injected. We only need the `mlx/nn/layers/`
# subtree, not the full turboquant_plus source tree (which would shadow
# the venv's numpy on Python 3.12).
OMLX_TURBOQUANT_PERSISTENT="${HOME}/.omlx/turboquant-plus/mlx/nn/layers"
if [[ -f "${OMLX_TURBOQUANT_PERSISTENT}/turbo_kv_cache.py" ]]; then
    # The file lives at .../mlx/nn/layers/turbo_kv_cache.py, so the
    # importable path is the parent of `mlx/`, which is
    # `~/.omlx/turboquant-plus/`.
    OMLX_TURBOQUANT_ROOT="$(dirname "$(dirname "$(dirname "$OMLX_TURBOQUANT_PERSISTENT")")")"
    case ":${PYTHONPATH:-}:" in
        *":${OMLX_TURBOQUANT_ROOT}:"*) ;;
        *) export PYTHONPATH="${OMLX_TURBOQUANT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
fi

# 2) phenotype-omlx Python packages.
case ":${PYTHONPATH:-}:" in
    *":${PHENOTYPE_OMLX_HOME}/python:"*) ;;
    *) export PYTHONPATH="${PHENOTYPE_OMLX_HOME}/python${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# 3) Reference research repos (read-only symlinks expected).
for p in turboquant_plus JetSpec ssd; do
    if [[ -d "${REPOS_ROOT}/${p}" ]]; then
        case ":${PYTHONPATH:-}:" in
            *":${REPOS_ROOT}/${p}:"*) ;;
            *) export PYTHONPATH="${REPOS_ROOT}/${p}${PYTHONPATH:+:${PYTHONPATH}}" ;;
        esac
    fi
done
# LatentMAS and TiDAR both provide a top-level `models` module — they must
# not be on PYTHONPATH simultaneously. Use the `use-latentmas` / `use-tidar`
# aliases to opt in.

# 4) Rust perf-core Python bindings (built artifacts).
if [[ -d "${PHENOTYPE_OMLX_HOME}/python/ffi" ]]; then
    case ":${PYTHONPATH:-}:" in
        *":${PHENOTYPE_OMLX_HOME}/python/ffi:"*) ;;
        *) export PYTHONPATH="${PHENOTYPE_OMLX_HOME}/python/ffi${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
fi

# 5) Local CLI bin (omlx-research proxy) — prepended to PATH.
case ":${PATH:-}:" in
    *":${PHENOTYPE_OMLX_HOME}/cli/bin:"*) ;;
    *) export PATH="${PHENOTYPE_OMLX_HOME}/cli/bin:${PATH}" ;;
esac

# Activate the Python venv associated with the research stack (must come
# AFTER the PYTHONPATH manipulation so the venv's site-packages take
# priority over the OMLX framework on 3.12+).
if [[ -f "${REPOS_ROOT}/turboquant_plus/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPOS_ROOT}/turboquant_plus/.venv/bin/activate"
elif [[ -f "${PHENOTYPE_OMLX_HOME}/.venv314/bin/activate" ]]; then
    # Native fork environment: Python 3.14+ with MLX/mlx-lm wheels.
    # This is the primary runtime; the historical TurboQuant venv is only
    # retained as a source-compatible fallback when present.
    source "${PHENOTYPE_OMLX_HOME}/.venv314/bin/activate"
fi

export PHENOTYPE_OMLX_HOME REPOS_ROOT
export PHENOTYPE_OMLX_READY="${PHENOTYPE_OMLX_HOME}/scripts/phenotype-omlx-ready"

alias use-latentmas='export PYTHONPATH='"${REPOS_ROOT}"'/LatentMAS:${PYTHONPATH:-}'
alias use-tidar='export PYTHONPATH='"${REPOS_ROOT}"'/TiDAR:${PYTHONPATH:-}'

echo "phenotype-omlx env ready: ${PHENOTYPE_OMLX_HOME} (python ${ACTIVE_PY_MINOR:-?}, offline=${PHENOTYPE_OMLX_OFFLINE:-1})"

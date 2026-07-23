#!/usr/bin/env bash
# Thin operator entry: Portage/Harbor (+ optional Langfuse / LangSmith plugins).
# Do not invent new ad-hoc eval runners — extend Harbor tasks / JobConfig.
#
# Requires:
#   PORTAGE_ROOT  — path to portage checkout (worktree OK; needs harbor-langfuse on main)
#   LANGFUSE_*    — when --langfuse is passed (primary)
#   LANGSMITH_API_KEY — when --langsmith is passed (legacy)
#   OPENAI_BASE_URL — when --niah is passed (OpenAI-compatible omlx/MLX server)
#
# Usage:
#   export PORTAGE_ROOT=.../worktrees/portage/<topic>
#   bash scripts/evals/run_via_harbor.sh              # hello-world oracle
#   bash scripts/evals/run_via_harbor.sh --langfuse   # + harbor-langfuse plugin
#   bash scripts/evals/run_via_harbor.sh --langsmith  # + harbor-langsmith (legacy)
#   bash scripts/evals/run_via_harbor.sh --policy     # omlx Qwen3.5 policy task
#   bash scripts/evals/run_via_harbor.sh --niah       # NIAH via OPENAI_BASE_URL
#   bash scripts/evals/run_via_harbor.sh --turbo      # TurboQuant SSOT gate
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${HARBOR_OUT:-$ROOT/.runs/harbor-eval}"
HARBOR_ENV="${HARBOR_ENV:-apple-container}"
USE_LF=0
USE_LS=0
MODE="hello"

for arg in "$@"; do
  case "$arg" in
    --langfuse) USE_LF=1 ;;
    --langsmith) USE_LS=1 ;;
    --policy) MODE="policy" ;;
    --niah) MODE="niah" ;;
    --turbo) MODE="turbo" ;;
    --help|-h)
      sed -n '2,22p' "$0"
      exit 0
      ;;
  esac
done

if [[ -z "${PORTAGE_ROOT:-}" ]]; then
  echo "ERROR: PORTAGE_ROOT required (portage / Harbor). No hardcoded worktree paths." >&2
  exit 2
fi
if [[ ! -d "$PORTAGE_ROOT" ]]; then
  echo "ERROR: PORTAGE_ROOT not a directory: $PORTAGE_ROOT" >&2
  exit 2
fi
if [[ "$HARBOR_ENV" != "apple-container" ]]; then
  echo "ERROR: HARBOR_ENV=$HARBOR_ENV forbidden; use apple-container" >&2
  exit 2
fi

mkdir -p "$OUT"
case "$MODE" in
  hello) TASK="$PORTAGE_ROOT/examples/tasks/hello-world" ;;
  policy) TASK="$ROOT/evals/harbor/tasks/omlx-qwen35-policy" ;;
  niah)
    TASK="$ROOT/evals/harbor/tasks/omlx-niah-api-smoke"
    if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
      echo "ERROR: OPENAI_BASE_URL required for --niah" >&2
      echo "  Self-host any OpenAI-compatible server and point here, e.g.:" >&2
      echo "    mlx_lm.server --model \$OMLX_READY_MODEL --host 127.0.0.1 --port 8766" >&2
      echo "    export OPENAI_BASE_URL=http://127.0.0.1:8766/v1   # host dry-run" >&2
      echo "    export OPENAI_BASE_URL=http://host.containers.internal:8766/v1  # Harbor→host" >&2
      exit 2
    fi
    ;;
  turbo) TASK="$ROOT/evals/harbor/tasks/omlx-turbo-ssot" ;;
  *) echo "ERROR: unknown mode $MODE" >&2; exit 2 ;;
esac

PLUGIN_ARGS=()
PYPATH_EXTRA=()
if [[ "$USE_LF" -eq 1 ]]; then
  if [[ -z "${LANGFUSE_PUBLIC_KEY:-}" || -z "${LANGFUSE_SECRET_KEY:-}" ]]; then
    echo "ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY required with --langfuse" >&2
    exit 2
  fi
  if [[ ! -d "$PORTAGE_ROOT/packages/harbor-langfuse/src" ]]; then
    echo "ERROR: harbor-langfuse missing at $PORTAGE_ROOT/packages/harbor-langfuse" >&2
    echo "  Use portage main (PR #478+) or a worktree that vendors the package." >&2
    exit 2
  fi
  export LANGFUSE_BASE_URL="${LANGFUSE_BASE_URL:-https://us.cloud.langfuse.com}"
  export HARBOR_LANGFUSE_ENVIRONMENT="${HARBOR_LANGFUSE_ENVIRONMENT:-harbor}"
  export HARBOR_LANGFUSE_FAIL_FAST="${HARBOR_LANGFUSE_FAIL_FAST:-true}"
  PYPATH_EXTRA+=("$PORTAGE_ROOT/packages/harbor-langfuse/src")
  PLUGIN_ARGS+=(--plugin langfuse)
  echo "langfuse base=$LANGFUSE_BASE_URL env=$HARBOR_LANGFUSE_ENVIRONMENT (SSOT: config/langfuse_harbor_kpis.json)"
fi
if [[ "$USE_LS" -eq 1 ]]; then
  if [[ -z "${LANGSMITH_API_KEY:-}" ]]; then
    echo "ERROR: LANGSMITH_API_KEY required with --langsmith" >&2
    exit 2
  fi
  PYPATH_EXTRA+=("$PORTAGE_ROOT/packages/harbor-langsmith/src")
  # Named LangSmith props (SSOT: config/langsmith_harbor_kpis.json)
  export HARBOR_LANGSMITH_DATASET="${HARBOR_LANGSMITH_DATASET:-omlx-harbor-tasks}"
  export HARBOR_LANGSMITH_EXPERIMENT="${HARBOR_LANGSMITH_EXPERIMENT:-omlx-harbor-${MODE}}"
  export HARBOR_LANGSMITH_FAIL_FAST="${HARBOR_LANGSMITH_FAIL_FAST:-true}"
  PLUGIN_ARGS+=(--plugin langsmith)
  echo "langsmith dataset=$HARBOR_LANGSMITH_DATASET experiment=$HARBOR_LANGSMITH_EXPERIMENT"
fi
if [[ ${#PYPATH_EXTRA[@]} -gt 0 ]]; then
  joined="$(IFS=:; echo "${PYPATH_EXTRA[*]}")"
  export PYTHONPATH="${joined}${PYTHONPATH:+:$PYTHONPATH}"
fi

# Surface SSOT model for agents / NIAH API smoke
export OMLX_READY_MODEL="${OMLX_READY_MODEL:-$(PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" python3 -m omlx_research.smoke_models readiness)}"
export OPENAI_MODEL="${OPENAI_MODEL:-$OMLX_READY_MODEL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-omlx}"

AGENT_ENV_ARGS=()
if [[ "$MODE" == "niah" ]]; then
  # Oracle uses solution.env; also pass --ae so JobConfig overlays cannot drop URLs.
  AGENT_ENV_ARGS+=(
    --ae "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
    --ae "OPENAI_API_KEY=${OPENAI_API_KEY}"
    --ae "OPENAI_MODEL=${OPENAI_MODEL}"
    --ae "OMLX_READY_MODEL=${OMLX_READY_MODEL}"
  )
fi

cd "$PORTAGE_ROOT"
echo "harbor env=$HARBOR_ENV mode=$MODE task=$TASK model=$OMLX_READY_MODEL out=$OUT"
uv run harbor run \
  -e "$HARBOR_ENV" \
  -p "$TASK" \
  -a oracle \
  -n 1 \
  -y \
  -o "$OUT" \
  "${AGENT_ENV_ARGS[@]}" \
  "${PLUGIN_ARGS[@]}"

echo "done. artifacts: $OUT"

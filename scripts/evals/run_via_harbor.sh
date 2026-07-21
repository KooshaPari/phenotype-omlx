#!/usr/bin/env bash
# Thin operator entry: Portage/Harbor (+ optional LangSmith plugin).
# Do not invent new ad-hoc eval runners — extend Harbor tasks / JobConfig.
#
# Requires:
#   PORTAGE_ROOT  — path to portage-TEMP checkout (worktree OK)
#   LANGSMITH_API_KEY — when --langsmith is passed
#   OPENAI_BASE_URL — when --niah is passed (OpenAI-compatible omlx/MLX server)
#
# Usage:
#   export PORTAGE_ROOT=.../worktrees/portage/<topic>
#   bash scripts/evals/run_via_harbor.sh              # hello-world oracle
#   bash scripts/evals/run_via_harbor.sh --langsmith  # + harbor-langsmith plugin
#   bash scripts/evals/run_via_harbor.sh --policy     # omlx Qwen3.5 policy task
#   bash scripts/evals/run_via_harbor.sh --niah       # NIAH via OPENAI_BASE_URL
#   bash scripts/evals/run_via_harbor.sh --turbo      # TurboQuant SSOT gate
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${HARBOR_OUT:-$ROOT/.runs/harbor-eval}"
HARBOR_ENV="${HARBOR_ENV:-apple-container}"
USE_LS=0
MODE="hello"

for arg in "$@"; do
  case "$arg" in
    --langsmith) USE_LS=1 ;;
    --policy) MODE="policy" ;;
    --niah) MODE="niah" ;;
    --turbo) MODE="turbo" ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

if [[ -z "${PORTAGE_ROOT:-}" ]]; then
  echo "ERROR: PORTAGE_ROOT required (portage-TEMP / Harbor). No hardcoded worktree paths." >&2
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
if [[ "$USE_LS" -eq 1 ]]; then
  if [[ -z "${LANGSMITH_API_KEY:-}" ]]; then
    echo "ERROR: LANGSMITH_API_KEY required with --langsmith" >&2
    exit 2
  fi
  export PYTHONPATH="$PORTAGE_ROOT/packages/harbor-langsmith/src${PYTHONPATH:+:$PYTHONPATH}"
  # Named LangSmith props (SSOT: config/langsmith_harbor_kpis.json)
  export HARBOR_LANGSMITH_DATASET="${HARBOR_LANGSMITH_DATASET:-omlx-harbor-tasks}"
  export HARBOR_LANGSMITH_EXPERIMENT="${HARBOR_LANGSMITH_EXPERIMENT:-omlx-harbor-${MODE}}"
  export HARBOR_LANGSMITH_FAIL_FAST="${HARBOR_LANGSMITH_FAIL_FAST:-true}"
  PLUGIN_ARGS=(--plugin langsmith)
  echo "langsmith dataset=$HARBOR_LANGSMITH_DATASET experiment=$HARBOR_LANGSMITH_EXPERIMENT"
fi

# Surface SSOT model for agents / NIAH API smoke
export OMLX_READY_MODEL="${OMLX_READY_MODEL:-$(PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" python3 -m omlx_research.smoke_models readiness)}"
export OPENAI_MODEL="${OPENAI_MODEL:-$OMLX_READY_MODEL}"

cd "$PORTAGE_ROOT"
echo "harbor env=$HARBOR_ENV mode=$MODE task=$TASK model=$OMLX_READY_MODEL out=$OUT"
uv run harbor run \
  -e "$HARBOR_ENV" \
  -p "$TASK" \
  -a oracle \
  -n 1 \
  -y \
  -o "$OUT" \
  "${PLUGIN_ARGS[@]}"

echo "done. artifacts: $OUT"

#!/usr/bin/env bash
# Thin operator entry: Portage/Harbor + **required** Langfuse plugin.
# Langfuse is the canonical observability backend (non-optional).
# LangSmith is removed from this path — do not reintroduce.
#
# Requires:
#   PORTAGE_ROOT              — path to portage-TEMP checkout (worktree OK)
#   LANGFUSE_PUBLIC_KEY       — always required
#   LANGFUSE_SECRET_KEY       — always required
#   OPENAI_BASE_URL           — when --niah is passed (OpenAI-compatible omlx/MLX)
#
# Usage:
#   export PORTAGE_ROOT=.../worktrees/portage/<topic>
#   export LANGFUSE_PUBLIC_KEY=pk-lf-...
#   export LANGFUSE_SECRET_KEY=sk-lf-...
#   export LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # optional
#   bash scripts/evals/run_via_harbor.sh              # hello-world oracle + Langfuse
#   bash scripts/evals/run_via_harbor.sh --policy
#   bash scripts/evals/run_via_harbor.sh --niah
#   bash scripts/evals/run_via_harbor.sh --turbo
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${HARBOR_OUT:-$ROOT/.runs/harbor-eval}"
HARBOR_ENV="${HARBOR_ENV:-apple-container}"
MODE="hello"

for arg in "$@"; do
  case "$arg" in
    --langsmith|--plugin-langsmith)
      echo "ERROR: LangSmith is removed from the operator path." >&2
      echo "  Canonical observability is Langfuse (harbor-langfuse)." >&2
      echo "  If Portage/Langfuse is buggy: fix Portage — do not fall back to LangSmith." >&2
      exit 2
      ;;
    --langfuse)
      # Accepted for back-compat; Langfuse is always on.
      ;;
    --policy) MODE="policy" ;;
    --niah) MODE="niah" ;;
    --turbo) MODE="turbo" ;;
    --help|-h)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $arg" >&2
      exit 2
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
if [[ -z "${LANGFUSE_PUBLIC_KEY:-}" || -z "${LANGFUSE_SECRET_KEY:-}" ]]; then
  echo "ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required (Langfuse is canonical)." >&2
  exit 2
fi
if [[ ! -d "$PORTAGE_ROOT/packages/harbor-langfuse/src" ]]; then
  echo "ERROR: harbor-langfuse missing under PORTAGE_ROOT (expected packages/harbor-langfuse)." >&2
  echo "  Fix Portage — LangSmith is not an allowed fallback." >&2
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
      echo "    mlx_lm server --model \$OMLX_READY_MODEL --host 127.0.0.1 --port 8766" >&2
      echo "    export OPENAI_BASE_URL=http://127.0.0.1:8766/v1   # host dry-run" >&2
      echo "    export OPENAI_BASE_URL=http://host.containers.internal:8766/v1  # Harbor→host" >&2
      exit 2
    fi
    ;;
  turbo) TASK="$ROOT/evals/harbor/tasks/omlx-turbo-ssot" ;;
  *) echo "ERROR: unknown mode $MODE" >&2; exit 2 ;;
esac

export LANGFUSE_BASE_URL="${LANGFUSE_BASE_URL:-${LANGFUSE_HOST:-https://us.cloud.langfuse.com}}"
export OBSERVABILITY_BACKEND=langfuse
export PYTHONPATH="$PORTAGE_ROOT/packages/harbor-langfuse/src${PYTHONPATH:+:$PYTHONPATH}"
PLUGIN_ARGS=(--plugin langfuse)
echo "langfuse base=$LANGFUSE_BASE_URL session=harbor_job_id (canonical)"

# Surface SSOT model for agents / NIAH API smoke
export OMLX_READY_MODEL="${OMLX_READY_MODEL:-$(PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" python3 -m omlx_research.smoke_models readiness)}"
export OPENAI_MODEL="${OPENAI_MODEL:-$OMLX_READY_MODEL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-omlx}"

AGENT_ENV_ARGS=()
if [[ "$MODE" == "niah" ]]; then
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

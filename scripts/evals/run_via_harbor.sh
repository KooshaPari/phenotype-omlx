#!/usr/bin/env bash
# Thin operator entry: Portage/Harbor (+ optional LangSmith plugin).
# Do not invent new ad-hoc eval runners — extend Harbor tasks / JobConfig.
#
# Requires:
#   PORTAGE_ROOT  — path to portage-TEMP checkout (worktree OK)
#   LANGSMITH_API_KEY — when --langsmith is passed
#
# Usage:
#   export PORTAGE_ROOT=.../worktrees/portage/<topic>
#   bash scripts/evals/run_via_harbor.sh              # hello-world oracle
#   bash scripts/evals/run_via_harbor.sh --langsmith  # + harbor-langsmith plugin
#   bash scripts/evals/run_via_harbor.sh --policy     # omlx Qwen3.5 policy task
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${HARBOR_OUT:-$ROOT/.runs/harbor-eval}"
HARBOR_ENV="${HARBOR_ENV:-apple-container}"
USE_LS=0
USE_POLICY=0

for arg in "$@"; do
  case "$arg" in
    --langsmith) USE_LS=1 ;;
    --policy) USE_POLICY=1 ;;
    --help|-h)
      sed -n '2,16p' "$0"
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
TASK="$PORTAGE_ROOT/examples/tasks/hello-world"
if [[ "$USE_POLICY" -eq 1 ]]; then
  TASK="$ROOT/evals/harbor/tasks/omlx-qwen35-policy"
fi

PLUGIN_ARGS=()
if [[ "$USE_LS" -eq 1 ]]; then
  if [[ -z "${LANGSMITH_API_KEY:-}" ]]; then
    echo "ERROR: LANGSMITH_API_KEY required with --langsmith" >&2
    exit 2
  fi
  export PYTHONPATH="$PORTAGE_ROOT/packages/harbor-langsmith/src${PYTHONPATH:+:$PYTHONPATH}"
  export HARBOR_LANGSMITH_DATASET="${HARBOR_LANGSMITH_DATASET:-omlx-harbor}"
  export HARBOR_LANGSMITH_EXPERIMENT="${HARBOR_LANGSMITH_EXPERIMENT:-omlx-eval}"
  PLUGIN_ARGS=(--plugin langsmith)
fi

# Surface SSOT model for agents that read OPENAI_MODEL / OMLX_READY_MODEL
export OMLX_READY_MODEL="${OMLX_READY_MODEL:-$(PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" python3 -m omlx_research.smoke_models readiness)}"
export OPENAI_MODEL="${OPENAI_MODEL:-$OMLX_READY_MODEL}"

cd "$PORTAGE_ROOT"
echo "harbor env=$HARBOR_ENV task=$TASK model=$OMLX_READY_MODEL out=$OUT"
uv run harbor run \
  -e "$HARBOR_ENV" \
  -p "$TASK" \
  -a oracle \
  -n 1 \
  -y \
  -o "$OUT" \
  "${PLUGIN_ARGS[@]}"

echo "done. artifacts: $OUT"

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
#   export PHENO_EXECUTION_WINDOW_ID=qwen35-20260802T0700Z-3090ti
#   bash scripts/evals/run_via_harbor.sh              # hello-world oracle + Langfuse
#   bash scripts/evals/run_via_harbor.sh --policy
#   bash scripts/evals/run_via_harbor.sh --niah
#   bash scripts/evals/run_via_harbor.sh --niah-8192
#   bash scripts/evals/run_via_harbor.sh --turbo
#   bash scripts/evals/run_via_harbor.sh --preflight --niah-8192
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${HARBOR_OUT:-$ROOT/.runs/harbor-eval}"
HARBOR_ENV="${HARBOR_ENV:-apple-container}"
MODE="hello"
PREFLIGHT=0

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
    --niah-8192) MODE="niah_8192" ;;
    --turbo) MODE="turbo" ;;
    --preflight) PREFLIGHT=1 ;;
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
if [[ -z "${PHENO_EXECUTION_WINDOW_ID:-}" ]]; then
  echo "ERROR: PHENO_EXECUTION_WINDOW_ID required; Harbor workloads need a bounded authorization window." >&2
  echo "  Set an operator-issued window ID before invoking Portage/Apple Container." >&2
  exit 2
fi
if [[ ! "${PHENO_EXECUTION_WINDOW_ID}" =~ ^[A-Za-z0-9._:-]{8,128}$ ]]; then
  echo "ERROR: PHENO_EXECUTION_WINDOW_ID must match [A-Za-z0-9._:-]{8,128}." >&2
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
# Harbor execution depends on the checked-out Portage source. Reject a path
# that cannot identify a committed revision, or one whose tracked revision
# still contains merge-conflict markers, before contacting Apple Container.
if ! git -C "$PORTAGE_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "ERROR: PORTAGE_ROOT must be a Git checkout with a valid HEAD." >&2
  exit 2
fi
if git -C "$PORTAGE_ROOT" grep -I -q -E '^(<<<<<<<|=======|>>>>>>>)' HEAD -- .; then
  echo "ERROR: PORTAGE_ROOT tracked HEAD contains an unresolved conflict marker." >&2
  exit 2
fi
mkdir -p "$OUT"
case "$MODE" in
  hello) TASK="$PORTAGE_ROOT/examples/tasks/hello-world" ;;
  policy) TASK="$ROOT/evals/harbor/tasks/omlx-qwen35-policy" ;;
  niah|niah_8192)
    TASK="$ROOT/evals/harbor/tasks/omlx-niah-api-smoke"
    if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
      # Apple Container often cannot resolve host.containers.internal.
      # Prefer a routable host LAN IP (MLX must listen on 0.0.0.0).
      HOST_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
      if [[ -n "$HOST_IP" ]]; then
        export OPENAI_BASE_URL="http://${HOST_IP}:8766/v1"
        echo "OPENAI_BASE_URL defaulted to $OPENAI_BASE_URL (host LAN)"
      else
        echo "ERROR: OPENAI_BASE_URL required for --niah" >&2
        echo "  Self-host any OpenAI-compatible server and point here, e.g.:" >&2
        echo "    mlx_lm server --model \$OMLX_READY_MODEL --host 0.0.0.0 --port 8766" >&2
        echo "    export OPENAI_BASE_URL=http://\$(ipconfig getifaddr en0):8766/v1" >&2
        exit 2
      fi
    fi
    ;;
  turbo) TASK="$ROOT/evals/harbor/tasks/omlx-turbo-ssot" ;;
  *) echo "ERROR: unknown mode $MODE" >&2; exit 2 ;;
esac

export LANGFUSE_BASE_URL="${LANGFUSE_BASE_URL:-${LANGFUSE_HOST:-https://us.cloud.langfuse.com}}"
export OBSERVABILITY_BACKEND=langfuse
export PYTHONPATH="$PORTAGE_ROOT/packages/harbor-langfuse/src${PYTHONPATH:+:$PYTHONPATH}"
# Prefer short name when entry-point installed; fall back to import path (PYTHONPATH).
PLUGIN_ARGS=(--plugin harbor_langfuse:LangfusePlugin)
echo "langfuse base=$LANGFUSE_BASE_URL session=harbor_job_id (canonical)"

# Surface SSOT model for agents / NIAH API smoke
export OMLX_READY_MODEL="${OMLX_READY_MODEL:-$(PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}" python3 -m omlx_research.smoke_models readiness)}"
export OPENAI_MODEL="${OPENAI_MODEL:-$OMLX_READY_MODEL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-omlx}"
case "$OMLX_READY_MODEL" in
  *Qwen2.5*|*qwen2.5*)
    echo "ERROR: Qwen2.5 is quarantined; Harbor requires Qwen3.5." >&2
    exit 2
    ;;
  *Qwen3.5*|*qwen3.5*)
    ;;
  *)
    echo "ERROR: Harbor model must be Qwen3.5 (got $OMLX_READY_MODEL)." >&2
    exit 2
    ;;
esac
if [[ "$OPENAI_MODEL" != "$OMLX_READY_MODEL" ]]; then
  echo "ERROR: OPENAI_MODEL must exactly match OMLX_READY_MODEL for provenance." >&2
  exit 2
fi

if [[ "$PREFLIGHT" -eq 1 ]]; then
  echo "preflight ok: env=$HARBOR_ENV mode=$MODE task=$TASK model=$OMLX_READY_MODEL window=$PHENO_EXECUTION_WINDOW_ID"
  echo "preflight only: Apple Container and Harbor were not invoked"
  exit 0
fi

# Apple Container's apiserver is an explicit user service, not a daemon that
# Harbor may assume is present. Start/verify it before creating any trials so
# XPC failures are diagnosed at the boundary.
source "$ROOT/scripts/evals/apple_container_preflight.sh"
ensure_apple_container_service

AGENT_ENV_ARGS=(--ae "PHENO_EXECUTION_WINDOW_ID=${PHENO_EXECUTION_WINDOW_ID}")
if [[ "$MODE" == "niah" || "$MODE" == "niah_8192" ]]; then
  AGENT_ENV_ARGS+=(
    --ae "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
    --ae "OPENAI_API_KEY=${OPENAI_API_KEY}"
    --ae "OPENAI_MODEL=${OPENAI_MODEL}"
    --ae "OMLX_READY_MODEL=${OMLX_READY_MODEL}"
  )
fi
if [[ "$MODE" == "niah_8192" ]]; then
  export NIAH_CONTEXT_TOKENS=8192
  AGENT_ENV_ARGS+=(--ae "NIAH_CONTEXT_TOKENS=8192")
fi

cd "$PORTAGE_ROOT"
echo "harbor env=$HARBOR_ENV mode=$MODE task=$TASK model=$OMLX_READY_MODEL out=$OUT"
# Empty-array expand is unsafe under `set -u` on some bash builds.
uv run harbor run \
  -e "$HARBOR_ENV" \
  -p "$TASK" \
  -a oracle \
  -n 1 \
  -y \
  -o "$OUT" \
  ${AGENT_ENV_ARGS[@]+"${AGENT_ENV_ARGS[@]}"} \
  "${PLUGIN_ARGS[@]}"

echo "done. artifacts: $OUT"

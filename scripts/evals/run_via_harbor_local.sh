#!/usr/bin/env bash
# Explicit local-only Harbor NIAH evaluation.  This never loads an exporter.
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${HARBOR_LOCAL_OUT:-$ROOT/.runs/harbor-local}"
HARBOR_ENV="${HARBOR_ENV:-apple-container}"
HARBOR_UV_BIN="${HARBOR_UV_BIN:-uv}"
HARBOR_PYTHON_BIN="${HARBOR_PYTHON_BIN:-python3}"

if [[ $# -ne 1 || "$1" != "--niah" ]]; then
  echo "usage: $0 --niah" >&2
  exit 2
fi
if [[ -z "${PORTAGE_ROOT:-}" || ! -d "$PORTAGE_ROOT" ]]; then
  echo "ERROR: PORTAGE_ROOT must name a Portage worktree" >&2
  exit 2
fi
if [[ "$HARBOR_ENV" != "apple-container" ]]; then
  echo "ERROR: HARBOR_ENV must be apple-container" >&2
  exit 2
fi
if [[ -z "${OMLX_READY_MODEL:-}" || "${OMLX_READY_MODEL,,}" != *qwen3.5* ]]; then
  echo "ERROR: OMLX_READY_MODEL must be an explicit Qwen3.5 model" >&2
  exit 2
fi
if [[ "${OPENAI_BASE_URL:-}" != http*":8766/v1" ]]; then
  echo "ERROR: OPENAI_BASE_URL must target the dedicated Qwen3.5 Harbor adapter on :8766/v1" >&2
  exit 2
fi

# This path is intentionally neither a fallback nor a remote observability run.
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY LANGFUSE_BASE_URL LANGFUSE_HOST OBSERVABILITY_BACKEND
export OPENAI_MODEL="${OPENAI_MODEL:-$OMLX_READY_MODEL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-omlx}"
mkdir -p "$OUT"

TASK="$ROOT/evals/harbor/tasks/omlx-niah-api-smoke"
cd "$PORTAGE_ROOT"
"$HARBOR_UV_BIN" run harbor run -e "$HARBOR_ENV" -p "$TASK" -a oracle -n 1 -y -o "$OUT" \
  --ae "OPENAI_BASE_URL=$OPENAI_BASE_URL" --ae "OPENAI_API_KEY=$OPENAI_API_KEY" \
  --ae "OPENAI_MODEL=$OPENAI_MODEL" --ae "OMLX_READY_MODEL=$OMLX_READY_MODEL"

# Keep Harbor's result.json immutable; emit a separately named, local report.
"$HARBOR_PYTHON_BIN" "$ROOT/scripts/evals/harbor_local_provenance.py" "$OUT" \
  --model "$OMLX_READY_MODEL" --output "$OUT/evaluation_report.local.json"
echo "done. local-only artifacts: $OUT"

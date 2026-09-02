#!/usr/bin/env bash
# Shell contract: invalid local Qwen3.5 requests never reach Harbor.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/portage"

if PORTAGE_ROOT="$TMP/portage" HARBOR_ENV="apple-container" \
  HARBOR_UV_BIN="$TMP/harbor-must-not-run" \
  OMLX_READY_MODEL="qWeN3.5-0.8B" \
  OPENAI_BASE_URL="http://host.docker.internal:8766/v1" \
  /bin/bash "$ROOT/scripts/evals/run_via_harbor_local.sh" --niah; then
  echo "invalid Qwen3.5 model unexpectedly reached Harbor" >&2
  exit 1
fi

if PORTAGE_ROOT="$TMP/portage" HARBOR_ENV="apple-container" \
  HARBOR_UV_BIN="$TMP/harbor-must-not-run" \
  OMLX_READY_MODEL="mlx-community/Qwen3.5-0.8B-OptiQ-4bit" \
  OPENAI_BASE_URL="http://127.0.0.1:8766/v1" \
  /bin/bash "$ROOT/scripts/evals/run_via_harbor_local.sh" --niah; then
  echo "invalid Qwen3.5 endpoint unexpectedly reached Harbor" >&2
  exit 1
fi

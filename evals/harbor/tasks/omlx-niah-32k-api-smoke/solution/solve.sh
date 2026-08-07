#!/usr/bin/env bash
# Oracle: NIAH via OpenAI-compatible API (OPENAI_BASE_URL required).
set -euo pipefail
export OMLX_NIAH_OUT=/app/niah_result.json
SCRIPT="$(dirname "$0")/niah_openai_smoke.py"
if [[ ! -f "$SCRIPT" ]]; then
  SCRIPT=/app/niah_openai_smoke.py
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "error: niah_openai_smoke.py not found next to solve.sh or under /app" >&2
  exit 2
fi
set +e
python3 "$SCRIPT"
rc=$?
set -e
if [[ -f /app/niah_result.json ]]; then
  cat /app/niah_result.json
fi
exit "$rc"

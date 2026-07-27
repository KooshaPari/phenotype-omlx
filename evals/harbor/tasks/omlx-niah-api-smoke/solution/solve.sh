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
python3 "$SCRIPT"
test -f /app/niah_answer.txt
grep -q '42-alpha' /app/niah_answer.txt
# Persist the structured request/usage contract in Harbor's oracle transcript.
cat /app/niah_result.json

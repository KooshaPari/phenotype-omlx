#!/bin/bash
# Harbor verifier: NIAH exact match + reward.txt.
set -euo pipefail
mkdir -p /logs/verifier
reward=0
if [[ -f /app/niah_answer.txt ]] && grep -q '42-alpha' /app/niah_answer.txt \
  && [[ -f /app/niah_result.json ]]; then
  if python3 - <<'PY'
import json, sys
d=json.load(open("/app/niah_result.json"))
sys.exit(0 if d.get("exact_match") and "qwen3.5" in d.get("model","").lower() else 1)
PY
  then
    reward=1
  fi
fi
echo "$reward" > /logs/verifier/reward.txt
exit $((1 - reward))

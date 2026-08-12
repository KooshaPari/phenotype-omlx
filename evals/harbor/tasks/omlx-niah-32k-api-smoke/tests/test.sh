#!/bin/bash
# Harbor verifier: NIAH 32k exact match + reward.txt.
# Verifier accepts any prompt_tokens that match the requested context (32k).
set -euo pipefail
mkdir -p /logs/verifier
reward=0
if [[ -f /app/niah_answer.txt ]] && grep -q '42-alpha' /app/niah_answer.txt \
  && [[ -f /app/niah_result.json ]]; then
  if python3 - <<'PY'
import json, sys
d = json.load(open("/app/niah_result.json"))
model = d.get("model")
ok = d.get("exact_match") is True and isinstance(model, str) and "qwen3.5" in model.lower()
# 32k contract: prompt_tokens == requested_context_tokens == 32768.
ok = ok and d.get("requested_context_tokens") == 32768
ok = ok and d.get("prompt_tokens") == 32768
ok = ok and d.get("context_tokens_exact") is True
sys.exit(0 if ok else 1)
PY
  then
    reward=1
  fi
fi
echo "$reward" > /logs/verifier/reward.txt
exit $((1 - reward))

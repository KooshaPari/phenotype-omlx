#!/bin/bash
# Harbor verifier: SSOT marker + reward.txt (required by Harbor).
set -euo pipefail
mkdir -p /logs/verifier
reward=0
if grep -qx 'turbo-qwen35-ssot-ok' /app/turbo_ssot_ok.txt 2>/dev/null; then
  reward=1
fi
echo "$reward" > /logs/verifier/reward.txt
exit $((1 - reward))

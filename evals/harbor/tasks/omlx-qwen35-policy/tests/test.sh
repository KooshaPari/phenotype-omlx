#!/bin/bash
# Harbor verifier: Qwen3.5 policy marker + reward.txt.
set -euo pipefail
mkdir -p /logs/verifier
reward=0
if grep -qx 'qwen35-ssot-ok' /app/policy_ok.txt 2>/dev/null; then
  reward=1
fi
echo "$reward" > /logs/verifier/reward.txt
exit $((1 - reward))

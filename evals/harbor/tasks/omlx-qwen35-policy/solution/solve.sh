#!/usr/bin/env bash
# Oracle: stamp policy marker (SSOT enforcement lives in repo CI / smoke_models).
set -euo pipefail
echo -n 'qwen35-ssot-ok' > /app/policy_ok.txt

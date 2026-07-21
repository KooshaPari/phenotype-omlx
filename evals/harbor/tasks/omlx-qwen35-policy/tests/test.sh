#!/usr/bin/env bash
set -euo pipefail
test -f /app/policy_ok.txt
grep -qx 'qwen35-ssot-ok' /app/policy_ok.txt

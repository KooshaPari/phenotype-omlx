#!/usr/bin/env bash
# Contract test for the NIAH verifier's exact Qwen3.5/8192 gate.
set -euo pipefail

TEST_ROOT="$(mktemp -d -t niah_task_contract.XXXXXX)"
trap 'rm -rf "${TEST_ROOT}"' EXIT

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_TEST="${ROOT}/evals/harbor/tasks/omlx-niah-api-smoke/tests/test.sh"
APP_ROOT="${TEST_ROOT}/app"
mkdir -p "${APP_ROOT}"
cp "${TASK_TEST}" "${TEST_ROOT}/test.sh"
sed -i '' "s#/app#${APP_ROOT}#g; s#/logs#${TEST_ROOT}/logs#g" "${TEST_ROOT}/test.sh"
chmod +x "${TEST_ROOT}/test.sh"
printf '%s\n' '42-alpha' >"${APP_ROOT}/niah_answer.txt"

cat >"${APP_ROOT}/niah_result.json" <<'EOF'
{"exact_match":"true","model":"Qwen/Qwen3.5-0.8B","requested_context_tokens":8192,"prompt_tokens":8192,"context_tokens_exact":true}
EOF
if "${TEST_ROOT}/test.sh"; then
  echo '[test_niah_task_contract] non-boolean exact_match was incorrectly accepted' >&2
  exit 1
fi

cat >"${APP_ROOT}/niah_result.json" <<'EOF'
{"exact_match":true,"model":"Qwen/Qwen3.5-0.8B","requested_context_tokens":0,"prompt_tokens":0,"context_tokens_exact":false}
EOF
if "${TEST_ROOT}/test.sh"; then
  echo '[test_niah_task_contract] zero-context result was incorrectly accepted' >&2
  exit 1
fi

cat >"${APP_ROOT}/niah_result.json" <<'EOF'
{"exact_match":true,"model":"Qwen/Qwen3.5-0.8B","requested_context_tokens":8192,"prompt_tokens":8192,"context_tokens_exact":true}
EOF
"${TEST_ROOT}/test.sh"
[[ "$(cat "${TEST_ROOT}/logs/verifier/reward.txt")" == 1 ]]
printf '%s\n' '[test_niah_task_contract] ok'

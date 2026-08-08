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

TASK_32K_ROOT="${ROOT}/evals/harbor/tasks/omlx-niah-32k-api-smoke"
TASK_32K_INSTRUCTION="${TASK_32K_ROOT}/instruction.md"
TASK_32K_JOB="${ROOT}/evals/harbor/jobs/niah-qwen35-local-32k.yaml"
if grep -q 'NIAH_CONTEXT_TOKENS_32K' "${TASK_32K_INSTRUCTION}" \
  || grep -q 'NIAH_CONTEXT_TOKENS_32K' "${TASK_32K_JOB}"; then
  echo '[test_niah_task_contract] stale 32k environment name remains' >&2
  exit 1
fi
grep -q 'NIAH_CONTEXT_TOKENS=32768' "${TASK_32K_INSTRUCTION}"
grep -q 'NIAH_CONTEXT_TOKENS: "32768"' "${TASK_32K_JOB}"
python3 - "${TASK_32K_ROOT}/task.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    task = tomllib.load(stream)

for scope in ("environment", "solution", "agent"):
    value = task[scope]["env"].get("NIAH_CONTEXT_TOKENS")
    if value != "32768":
        raise SystemExit(f"{scope}.env NIAH_CONTEXT_TOKENS must be literal 32768")
PY

TASK_TEST="${TASK_32K_ROOT}/tests/test.sh"
APP_ROOT_32K="${TEST_ROOT}/app-32k"
mkdir -p "${APP_ROOT_32K}"
cp "${TASK_TEST}" "${TEST_ROOT}/test-32k.sh"
sed -i '' "s#/app#${APP_ROOT_32K}#g; s#/logs#${TEST_ROOT}/logs-32k#g" \
  "${TEST_ROOT}/test-32k.sh"
chmod +x "${TEST_ROOT}/test-32k.sh"
printf '%s\n' '42-alpha' >"${APP_ROOT_32K}/niah_answer.txt"

cat >"${APP_ROOT_32K}/niah_result.json" <<'EOF'
{"exact_match":true,"model":"Qwen/Qwen3.5-0.8B","requested_context_tokens":8192,"prompt_tokens":8192,"context_tokens_exact":true}
EOF
if "${TEST_ROOT}/test-32k.sh"; then
  echo '[test_niah_task_contract] 8k result was incorrectly accepted by 32k verifier' >&2
  exit 1
fi

cat >"${APP_ROOT_32K}/niah_result.json" <<'EOF'
{"exact_match":true,"model":"Qwen/Qwen3.5-0.8B","requested_context_tokens":32768,"prompt_tokens":32768,"context_tokens_exact":true}
EOF
"${TEST_ROOT}/test-32k.sh"
[[ "$(cat "${TEST_ROOT}/logs-32k/verifier/reward.txt")" == 1 ]]

printf '%s\n' '[test_niah_task_contract] ok'

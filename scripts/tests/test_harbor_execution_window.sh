#!/usr/bin/env bash
# Contract test: Harbor must reject absent or malformed execution windows
# before touching Portage, Apple Container, or the Harbor CLI.
set -euo pipefail

TEST_ROOT="$(mktemp -d -t harbor_execution_window.XXXXXX)"
trap 'rm -rf "${TEST_ROOT}"' EXIT

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${ROOT}/scripts/evals/run_via_harbor.sh"
CALL_LOG="${TEST_ROOT}/calls.log"
FAKE_BIN="${TEST_ROOT}/bin"
mkdir -p "${FAKE_BIN}"

cat >"${FAKE_BIN}/uv" <<'EOF'
#!/usr/bin/env bash
printf 'uv %s\n' "$*" >>"${HARBOR_WINDOW_CALL_LOG:?}"
exit 99
EOF
cat >"${FAKE_BIN}/container" <<'EOF'
#!/usr/bin/env bash
printf 'container %s\n' "$*" >>"${HARBOR_WINDOW_CALL_LOG:?}"
exit 99
EOF
chmod +x "${FAKE_BIN}/uv" "${FAKE_BIN}/container"

export HARBOR_WINDOW_CALL_LOG="${CALL_LOG}"
export PATH="${FAKE_BIN}:${PATH}"

set +e
PORTAGE_ROOT="${TEST_ROOT}/missing-portage" \
  LANGFUSE_PUBLIC_KEY=public \
  LANGFUSE_SECRET_KEY=secret \
  bash "${RUNNER}" --policy >"${TEST_ROOT}/missing.out" 2>"${TEST_ROOT}/missing.err"
missing_rc=$?
set -e

if [[ "${missing_rc}" -ne 2 ]]; then
  cat "${TEST_ROOT}/missing.err" >&2
  printf '[test_harbor_execution_window] expected missing-window exit 2, got %s\n' "${missing_rc}" >&2
  exit 1
fi
grep -q 'PHENO_EXECUTION_WINDOW_ID' "${TEST_ROOT}/missing.err"
[[ ! -s "${CALL_LOG}" ]]

set +e
PHENO_EXECUTION_WINDOW_ID='bad window' \
  PORTAGE_ROOT="${TEST_ROOT}/missing-portage" \
  LANGFUSE_PUBLIC_KEY=public \
  LANGFUSE_SECRET_KEY=secret \
  bash "${RUNNER}" --policy >"${TEST_ROOT}/malformed.out" 2>"${TEST_ROOT}/malformed.err"
malformed_rc=$?
set -e

if [[ "${malformed_rc}" -ne 2 ]]; then
  cat "${TEST_ROOT}/malformed.err" >&2
  printf '[test_harbor_execution_window] expected malformed-window exit 2, got %s\n' "${malformed_rc}" >&2
  exit 1
fi
grep -q 'PHENO_EXECUTION_WINDOW_ID' "${TEST_ROOT}/malformed.err"
[[ ! -s "${CALL_LOG}" ]]

CONFLICTED_PORTAGE="${TEST_ROOT}/conflicted-portage"
mkdir -p "${CONFLICTED_PORTAGE}/packages/harbor-langfuse/src"
git -C "${CONFLICTED_PORTAGE}" init -q
git -C "${CONFLICTED_PORTAGE}" config user.email 'fixture@example.invalid'
git -C "${CONFLICTED_PORTAGE}" config user.name 'Harbor fixture'
printf '%s\n' '<<<<<<< HEAD' 'fixture conflict' '=======' 'other fixture' '>>>>>>> topic' \
  >"${CONFLICTED_PORTAGE}/source.txt"
git -C "${CONFLICTED_PORTAGE}" add source.txt
git -C "${CONFLICTED_PORTAGE}" commit -qm 'fixture source with conflict marker'

set +e
PHENO_EXECUTION_WINDOW_ID='window-1234' \
  PORTAGE_ROOT="${CONFLICTED_PORTAGE}" \
  LANGFUSE_PUBLIC_KEY=public \
  LANGFUSE_SECRET_KEY=secret \
  bash "${RUNNER}" --policy >"${TEST_ROOT}/conflicted.out" 2>"${TEST_ROOT}/conflicted.err"
conflicted_rc=$?
set -e

if [[ "${conflicted_rc}" -ne 2 ]]; then
  cat "${TEST_ROOT}/conflicted.err" >&2
  printf '[test_harbor_execution_window] expected conflict-marker exit 2, got %s\n' \
    "${conflicted_rc}" >&2
  exit 1
fi
grep -q 'conflict marker' "${TEST_ROOT}/conflicted.err"
[[ ! -s "${CALL_LOG}" ]]

NONGIT_PORTAGE="${TEST_ROOT}/nongit-portage"
mkdir -p "${NONGIT_PORTAGE}/packages/harbor-langfuse/src"

set +e
PHENO_EXECUTION_WINDOW_ID='window-1234' \
  PORTAGE_ROOT="${NONGIT_PORTAGE}" \
  LANGFUSE_PUBLIC_KEY=public \
  LANGFUSE_SECRET_KEY=secret \
  bash "${RUNNER}" --policy >"${TEST_ROOT}/nongit.out" 2>"${TEST_ROOT}/nongit.err"
nongit_rc=$?
set -e

if [[ "${nongit_rc}" -ne 2 ]]; then
  cat "${TEST_ROOT}/nongit.err" >&2
  printf '[test_harbor_execution_window] expected non-Git-root exit 2, got %s\n' "${nongit_rc}" >&2
  exit 1
fi
grep -q 'Git checkout' "${TEST_ROOT}/nongit.err"
[[ ! -s "${CALL_LOG}" ]]

PREFLIGHT_PORTAGE="${TEST_ROOT}/preflight-portage"
mkdir -p "${PREFLIGHT_PORTAGE}/packages/harbor-langfuse/src"
git -C "${PREFLIGHT_PORTAGE}" init -q
git -C "${PREFLIGHT_PORTAGE}" config user.email 'fixture@example.invalid'
git -C "${PREFLIGHT_PORTAGE}" config user.name 'Harbor fixture'
printf '%s\n' 'clean fixture' >"${PREFLIGHT_PORTAGE}/source.txt"
git -C "${PREFLIGHT_PORTAGE}" add source.txt
git -C "${PREFLIGHT_PORTAGE}" commit -qm 'clean fixture source'

PHENO_EXECUTION_WINDOW_ID='window-1234' \
  PORTAGE_ROOT="${PREFLIGHT_PORTAGE}" \
  LANGFUSE_PUBLIC_KEY=public \
  LANGFUSE_SECRET_KEY=secret \
  OMLX_READY_MODEL='Qwen/Qwen3.5-0.8B' \
  bash "${RUNNER}" --preflight --policy >"${TEST_ROOT}/preflight.out" 2>"${TEST_ROOT}/preflight.err"
grep -q 'preflight ok' "${TEST_ROOT}/preflight.out"
grep -q 'were not invoked' "${TEST_ROOT}/preflight.out"
[[ ! -s "${CALL_LOG}" ]]

PHENO_EXECUTION_WINDOW_ID='window-1234' \
  PORTAGE_ROOT="${PREFLIGHT_PORTAGE}" \
  LANGFUSE_PUBLIC_KEY=public \
  LANGFUSE_SECRET_KEY=secret \
  OMLX_READY_MODEL='Qwen/Qwen3.5-0.8B' \
  OPENAI_BASE_URL='http://127.0.0.1:8766/v1' \
  bash "${RUNNER}" --preflight --niah-8192 >"${TEST_ROOT}/niah-preflight.out" 2>"${TEST_ROOT}/niah-preflight.err"
grep -q 'mode=niah_8192' "${TEST_ROOT}/niah-preflight.out"
grep -q 'context=8192' "${TEST_ROOT}/niah-preflight.out"
[[ ! -s "${CALL_LOG}" ]]

grep -q 'Qwen2\.5' "${RUNNER}"
grep -q 'OPENAI_MODEL' "${RUNNER}"

printf '%s\n' '[test_harbor_execution_window] ok'

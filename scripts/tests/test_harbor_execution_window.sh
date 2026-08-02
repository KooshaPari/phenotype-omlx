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

printf '%s\n' '[test_harbor_execution_window] ok'

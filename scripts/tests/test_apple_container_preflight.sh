#!/usr/bin/env bash
# Contract test for the Harbor Apple Container preflight.
set -euo pipefail

TEST_ROOT="$(mktemp -d -t apple_container_preflight.XXXXXX)"
trap 'rm -rf "$TEST_ROOT"' EXIT

HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/../evals" && pwd)/apple_container_preflight.sh"
FAKE_CONTAINER="$TEST_ROOT/container"
STATE="$TEST_ROOT/state"
LOG="$TEST_ROOT/log"

cat >"$FAKE_CONTAINER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-} ${2:-}" in
  "system status")
    if [[ -f "${APPLE_CONTAINER_TEST_STATE:?}" ]]; then
      printf 'FIELD VALUE\nstatus running\n'
    else
      printf 'FIELD VALUE\nstatus stopped\n'
    fi
    ;;
  "system start")
    printf 'start\n' >>"${APPLE_CONTAINER_TEST_LOG:?}"
    : >"${APPLE_CONTAINER_TEST_STATE:?}"
    ;;
  *)
    printf 'unexpected fake container args: %s\n' "$*" >&2
    exit 2
    ;;
esac
EOF
chmod +x "$FAKE_CONTAINER"

export APPLE_CONTAINER_TEST_STATE="$STATE"
export APPLE_CONTAINER_TEST_LOG="$LOG"
export CONTAINER_BIN="$FAKE_CONTAINER"

# The helper is expected to start a stopped service and verify the resulting state.
# This test intentionally fails until the helper exists and is sourced by the runner.
source "$HELPER"
ensure_apple_container_service

[[ "$(cat "$LOG")" == start ]]

printf '%s\n' '[test_apple_container_preflight] ok'

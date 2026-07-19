#!/usr/bin/env bash
# snapshot.sh — release-gate runner that wraps `airlock-v2 snapshot`.
#
# Airlock v2 has no `promote` subcommand. The project's promotion mechanism is
# `airlock-v2 snapshot`, which creates and pushes a `wip/<date>-<uuid>` branch.
# This script enforces the CI gates BEFORE invoking that snapshot, so a snapshot
# is only ever taken from a known-green commit.
#
# Gates (in order):
#   1. airlock-v2 on PATH                       -> exit 1
#   2. working tree clean                       -> exit 2
#   3. cargo test --workspace --all-targets     -> exit 3 (if any test fails)
#   4. cargo clippy --workspace -- -D warnings  -> exit 4
#   5. python3 -m pytest -q                     -> exit 5
#   6. python3 -m omlx_research.cli doctor      -> exit 6 (if any [FAIL] row)
#
# After all gates pass, `airlock-v2 snapshot` is invoked from the repo root.
# Idempotent: re-running creates a fresh wip/<date>-<uuid> each time.
#
# Usage:
#   bash scripts/snapshot.sh
#   DRY_RUN=1 bash scripts/snapshot.sh   # run gates, skip the actual snapshot
#
# Bash 3.2 baseline (macOS). No `[[ ]]`, no mapfile, no associative arrays.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

log() { printf '[snapshot] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Gate 1 — airlock-v2 on PATH
# ---------------------------------------------------------------------------
if ! command -v airlock-v2 >/dev/null 2>&1; then
    echo 'airlock-v2 not on PATH — run scripts/install_airlock_v2.sh first' >&2
    exit 1
fi
log "gate 1 ok: airlock-v2 ($(command -v airlock-v2))"

# ---------------------------------------------------------------------------
# Gate 2 — working tree clean
# ---------------------------------------------------------------------------
if [ -n "$(git status --porcelain)" ]; then
    echo 'working tree is dirty: commit or stash first' >&2
    git status --short >&2
    exit 2
fi
log "gate 2 ok: working tree clean on $(git rev-parse --abbrev-ref HEAD)"

# ---------------------------------------------------------------------------
# Gate 3 — Rust workspace tests
# ---------------------------------------------------------------------------
log "gate 3: running cargo test --workspace --all-targets..."
TEST_OUT="$(cd perf-core && cargo test --workspace --all-targets 2>&1)"
TEST_SUMMARY="$(printf '%s\n' "${TEST_OUT}" | grep -E '^test result' | \
    awk -F'[ .;]+' '{p+=$5; f+=$7; i+=$9} END {print "passed=" p, "failed=" f, "ignored=" i}')"
log "  ${TEST_SUMMARY}"
FAILED_COUNT="$(printf '%s\n' "${TEST_SUMMARY}" | \
    awk '{for(i=1;i<=NF;i++) if($i ~ /^failed=/) {split($i,kv,"="); print kv[2]}}')"
if [ -z "${FAILED_COUNT}" ]; then FAILED_COUNT=0; fi
if [ "${FAILED_COUNT}" -ne 0 ]; then
    echo "rust tests failed: ${FAILED_COUNT} (see output above)" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Gate 4 — clippy with -D warnings
# ---------------------------------------------------------------------------
log "gate 4: running cargo clippy --workspace --all-targets -- -D warnings..."
CLIPPY_OUT="$(cd perf-core && cargo clippy --workspace --all-targets -- -D warnings 2>&1)"
CLIPPY_RC=$?
if [ "${CLIPPY_RC}" -ne 0 ]; then
    printf '%s\n' "${CLIPPY_OUT}" >&2
    echo "cargo clippy -- -D warnings failed (rc=${CLIPPY_RC})" >&2
    exit 4
fi
log "  clippy clean"

# ---------------------------------------------------------------------------
# Gate 5 — Python pytest
# ---------------------------------------------------------------------------
log "gate 5: running python3 -m pytest -q..."
PYTEST_OUT="$(cd python && python3 -m pytest -q 2>&1)"
PYTEST_RC=$?
if [ "${PYTEST_RC}" -ne 0 ]; then
    printf '%s\n' "${PYTEST_OUT}" >&2
    echo "pytest reported failures (rc=${PYTEST_RC})" >&2
    exit 5
fi
log "  pytest clean"

# ---------------------------------------------------------------------------
# Gate 6 — omlx_research doctor (count [FAIL] rows)
# ---------------------------------------------------------------------------
# The doctor may return a non-zero rc when there are WARN rows (warnings
# escalate the rc), but the task spec is explicit: only `[FAIL]` rows block
# the snapshot. We deliberately ignore the doctor's own exit code and count
# FAIL rows ourselves. We also disable errexit for this substitution because
# some bash versions propagate the subshell's non-zero exit into the
# assignment under `set -euo pipefail`, even though the assignment itself
# does not fail.
log "gate 6: running python3 -m omlx_research.cli doctor..."
set +e
DOCTOR_OUT="$(cd python && python3 -m omlx_research.cli doctor 2>&1)"
DOCTOR_RC=$?
set -e
if [ "${DOCTOR_RC}" -ne 0 ]; then
    log "  doctor exited rc=${DOCTOR_RC} (treating WARN as non-fatal)"
fi
FAIL_COUNT="$(printf '%s\n' "${DOCTOR_OUT}" | grep -c -E '^\[FAIL\]' || true)"
if [ "${FAIL_COUNT}" != "0" ]; then
    echo "doctor reported ${FAIL_COUNT} FAIL check(s)" >&2
    printf '%s\n' "${DOCTOR_OUT}" | grep -E '^\[(FAIL|WARN|OK)' >&2 || true
    exit 6
fi
log "  doctor clean (${FAIL_COUNT} FAIL)"

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
if [ "${DRY_RUN:-0}" = "1" ]; then
    log "DRY RUN — would have called: airlock-v2 snapshot"
    exit 0
fi

log "all gates green — invoking airlock-v2 snapshot"
airlock-v2 snapshot

# After `airlock-v2 snapshot`, HEAD is detached on the new wip/<date>-<uuid>.
log "[snapshot] wip branch: $(git branch --show-current)"
exit 0
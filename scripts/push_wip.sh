#!/usr/bin/env bash
# push_wip.sh — push the WIP branch with airlock-v2 retry + exponential
# backoff, falling back to plain `git push` if airlock-v2 is not on PATH.
#
# Rationale
# ---------
# Turn-9 resume notes §5 flagged airlock-v2 push retries as a known
# P2 source of friction: a transient airlock-v2 server-side glitch
# leaves the local WIP branch one commit ahead of origin, and a
# naive `git push` aborts the workflow mid-session. This script
# wraps the push in exponential backoff so the same commit
# eventually lands without manual intervention.
#
# Behavior
# --------
# 1. If `airlock-v2 push` succeeds on the first try, exit 0.
# 2. On failure, retry with exponential backoff starting at
#    `INITIAL_BACKOFF_SECONDS` and doubling each step up to
#    `MAX_BACKOFF_SECONDS`. Total wait budget is bounded by
#    `MAX_TOTAL_SECONDS`.
# 3. After exhausting retries with airlock-v2, try a plain
#    `git push` (no preflight) so the same commit can still land
#    if the airlock-v2 server is completely offline.
# 4. If both fail, surface the last captured error and exit
#    non-zero so the user can intervene.
#
# Usage
# -----
#   bash scripts/push_wip.sh [<remote>] [<branch>]
#   bash scripts/push_wip.sh              # uses defaults: origin HEAD
#   bash scripts/push_wip.sh origin chore/my-wip-branch
#
# Environment
# -----------
#   AIRLOCK_DISABLE=1    — skip airlock-v2 entirely, use plain `git push`
#   AIRLOCK_VERBOSE=1    — log every retry attempt to stderr
#   INITIAL_BACKOFF_SECONDS=2 (default)
#   MAX_BACKOFF_SECONDS=60 (default)
#   MAX_TOTAL_SECONDS=120 (default)
#   MAX_RETRIES=3 (default; total attempts = MAX_RETRIES + 1)
#   PUSH_BIN=<path>      — override the airlock-v2 binary (for tests)
#   GIT_BIN=<path>       — override git (for tests)
#   PUSH_WIP_REPO_ROOT=<dir> — override the repo root (for tests); the
#                              working-tree-dirty check and subsequent
#                              push both operate on this directory.
#
# Exit codes
# ----------
#   0  push succeeded (via airlock-v2 or plain git push)
#   1  both airlock-v2 and plain push failed; details on stderr
#   2  repo state is dirty (uncommitted working-tree changes); push
#      was skipped to avoid accidentally shipping WIP edits

set -euo pipefail

if [[ -n "${PUSH_WIP_REPO_ROOT:-}" ]]; then
    REPO_ROOT="${PUSH_WIP_REPO_ROOT}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
REMOTE="${1:-origin}"
BRANCH="${2:-HEAD}"

# Defaults overridable via env. Defaults are deliberately small so
# automated CI invocations do not block for minutes.
INITIAL_BACKOFF_SECONDS="${INITIAL_BACKOFF_SECONDS:-2}"
MAX_BACKOFF_SECONDS="${MAX_BACKOFF_SECONDS:-60}"
MAX_TOTAL_SECONDS="${MAX_TOTAL_SECONDS:-120}"
MAX_RETRIES="${MAX_RETRIES:-3}"

: "${PUSH_BIN:=airlock-v2}"
: "${GIT_BIN:=git}"

_log() {
    if [[ "${AIRLOCK_VERBOSE:-0}" == "1" ]]; then
        printf '[push_wip] %s\n' "$*" >&2
    fi
}

_fail() {
    printf '[push_wip] ERROR: %s\n' "$*" >&2
    exit "${1:-1}"
}

# 1. Refuse to push if the working tree is dirty. The script's name
#    is `push_wip.sh`, not `push_anything.sh`; it should not be a
#    generic `git push` wrapper that auto-stashes arbitrary edits.
# Untracked files (?? prefix) are included so newly-added notebooks
# or research scratch files still gate the push.
if "${GIT_BIN}" -C "${REPO_ROOT}" status --porcelain >/dev/null 2>&1; then
    if "${GIT_BIN}" -C "${REPO_ROOT}" status --porcelain \
            | grep -E '^(.[MDU ?]|..[MDU])' >/dev/null; then
        _fail 2 "working tree dirty; commit or stash before push"
    fi
fi

_log "pushing ${REMOTE} ${BRANCH}"

# 2. Resolve branch name if HEAD was passed (handles detached HEAD).
if [[ "${BRANCH}" == "HEAD" ]]; then
    if BRANCH_RESOLVED="$("${GIT_BIN}" -C "${REPO_ROOT}" symbolic-ref --short HEAD 2>/dev/null)"; then
        BRANCH="${BRANCH_RESOLVED}"
    else
        _fail "HEAD is detached; pass an explicit branch name"
        exit 1
    fi
fi

# 3. Decide which transport to use: airlock-v2 unless disabled or
#    missing; plain git push otherwise. PUSH_BIN may be an absolute
#    path (test fixtures) or a bare command name (PATH lookup).
use_airlock=0
if [[ "${AIRLOCK_DISABLE:-0}" != "1" ]]; then
    if [[ "${PUSH_BIN}" == */* ]] && [[ -x "${PUSH_BIN}" ]]; then
        use_airlock=1
    elif command -v "${PUSH_BIN}" >/dev/null 2>&1; then
        use_airlock=1
    fi
fi

if [[ "${use_airlock}" == "1" ]]; then
    _log "using ${PUSH_BIN} transport"
else
    _log "using plain ${GIT_BIN} push (${PUSH_BIN} unavailable)"
fi

# 4. Push loop with exponential backoff.
succeeded=0
last_error=""
backoff="${INITIAL_BACKOFF_SECONDS}"
deadline=$(( $(date +%s) + MAX_TOTAL_SECONDS ))
attempt=0

while :; do
    attempt=$((attempt + 1))
    if [[ "${use_airlock}" == "1" ]]; then
        _log "attempt ${attempt}: ${PUSH_BIN} push ${REMOTE} ${BRANCH}"
        set +e
        err="$("${PUSH_BIN}" push "${REMOTE}" "${BRANCH}" 2>&1)"
        rc=$?
        set -e
        if (( rc == 0 )); then
            succeeded=1
            break
        fi
        last_error="${err}"
    else
        _log "attempt ${attempt}: ${GIT_BIN} push ${REMOTE} ${BRANCH}"
        set +e
        err="$("${GIT_BIN}" -C "${REPO_ROOT}" push "${REMOTE}" "${BRANCH}" 2>&1)"
        rc=$?
        set -e
        if (( rc == 0 )); then
            succeeded=1
            break
        fi
        last_error="${err}"
    fi

    # Stop conditions: exhausted retries OR deadline exceeded.
    now=$(date +%s)
    if (( now >= deadline )); then
        _log "deadline ${MAX_TOTAL_SECONDS}s exceeded; giving up on ${REMOTE}"
        break
    fi
    if (( attempt > MAX_RETRIES )); then
        _log "max retries (${MAX_RETRIES}) exceeded; giving up"
        break
    fi

    _log "retrying after ${backoff}s"
    sleep "${backoff}"
    backoff=$(( backoff * 2 ))
    if (( backoff > MAX_BACKOFF_SECONDS )); then
        backoff="${MAX_BACKOFF_SECONDS}"
    fi
done

if (( succeeded )); then
    exit 0
fi

# 5. Fallback: try plain git push if airlock-v2 was used and failed.
if [[ "${use_airlock}" == "1" ]]; then
    _log "airlock-v2 exhausted; falling back to plain ${GIT_BIN} push"
    if "${GIT_BIN}" -C "${REPO_ROOT}" push "${REMOTE}" "${BRANCH}" >/dev/null 2>&1; then
        exit 0
    fi
fi

_fail "all push attempts failed; last error: ${last_error:-<none captured>}"

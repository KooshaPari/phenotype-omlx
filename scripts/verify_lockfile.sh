#!/usr/bin/env bash
# verify_lockfile.sh — verify perf-core/Cargo.lock has not drifted from
# the SHA-256 fingerprint recorded in perf-core/lockfile.lock.
#
# Bash 3.2 portable (macOS default). No external deps beyond `shasum`,
# `awk`, and `grep` — all present on macOS and Linux by default.
#
# Usage:
#   bash scripts/verify_lockfile.sh
#
# Exit codes:
#   0  digest matches the recorded fingerprint
#   1  digest differs (lockfile has drifted) or fingerprint file is malformed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCKFILE="${REPO_ROOT}/perf-core/Cargo.lock"
FINGERPRINT="${REPO_ROOT}/perf-core/lockfile.lock"

if [[ ! -f "${LOCKFILE}" ]]; then
    echo "[lockfile] ERROR: missing ${LOCKFILE}" >&2
    exit 1
fi

if [[ ! -f "${FINGERPRINT}" ]]; then
    echo "[lockfile] ERROR: missing ${FINGERPRINT}" >&2
    exit 1
fi

actual="$(shasum -a 256 "${LOCKFILE}" | awk '{print $1}')"

# Parse the `sha256:` line from the fingerprint file (one-liner, awk only).
expected="$(awk -F': ' '/^sha256:[[:space:]]/{print $2; exit}' "${FINGERPRINT}")"

if [[ -z "${expected}" ]]; then
    echo "[lockfile] ERROR: could not parse sha256: line in ${FINGERPRINT}" >&2
    exit 1
fi

if [[ "${actual}" == "${expected}" ]]; then
    echo "[lockfile] OK: ${actual}"
    exit 0
fi

echo "[lockfile] MISMATCH expected=${expected} got=${actual}" >&2
exit 1
#!/usr/bin/env bash
# install_airlock_v2.sh — install the airlock-v2 binary to a directory on PATH.
#
# The Airlock v2 binary lives in the PhenoVCS workspace at
# `${PHENOTYPE_PHENOVCS_HOME:-/Users/kooshapari/CodeProjects/Phenotype/repos/PhenoVCS}`.
# This script builds (if needed) and symlinks it into a PATH directory so
# the `airlock-v2` subcommand is invokable from any shell.
#
# Usage:
#   bash scripts/install_airlock_v2.sh
#
# Idempotent: if the symlink already exists and points to a working
# binary, this script exits 0 without changes.

set -euo pipefail

PHENOVCS_HOME="${PHENOTYPE_PHENOVCS_HOME:-/Users/kooshapari/CodeProjects/Phenotype/repos/PhenoVCS}"
AIRLOCK_V2_BIN="${PHENOVCS_HOME}/target/release/airlock-v2"
AIRLOCK_V2_LINK_DIR="${AIRLOCK_V2_LINK_DIR:-/opt/homebrew/bin}"
AIRLOCK_V2_LINK="${AIRLOCK_V2_LINK_DIR}/airlock-v2"

# 1) Build the binary if missing.
if [[ ! -x "${AIRLOCK_V2_BIN}" ]]; then
    echo "[install_airlock_v2] Building airlock-v2 at ${AIRLOCK_V2_BIN}..."
    if [[ ! -f "${PHENOVCS_HOME}/Cargo.toml" ]]; then
        echo "[install_airlock_v2] ERROR: PhenoVCS workspace not found at ${PHENOVCS_HOME}" >&2
        echo "[install_airlock_v2] Set PHENOTYPE_PHENOVCS_HOME to the PhenoVCS repo root." >&2
        exit 1
    fi
    ( cd "${PHENOVCS_HOME}" && cargo build -p airlock-v2 --release )
fi

# 2) Verify the binary works.
if ! "${AIRLOCK_V2_BIN}" --version >/dev/null 2>&1; then
    echo "[install_airlock_v2] ERROR: ${AIRLOCK_V2_BIN} did not respond to --version" >&2
    exit 1
fi

# 3) Symlink into the link directory.
if [[ -L "${AIRLOCK_V2_LINK}" ]]; then
    current_target="$(readlink "${AIRLOCK_V2_LINK}")"
    if [[ "${current_target}" == "${AIRLOCK_V2_BIN}" ]]; then
        echo "[install_airlock_v2] Symlink already correct: ${AIRLOCK_V2_LINK} -> ${AIRLOCK_V2_BIN}"
        exit 0
    fi
    echo "[install_airlock_v2] Replacing stale symlink at ${AIRLOCK_V2_LINK} (-> ${current_target})"
    rm "${AIRLOCK_V2_LINK}"
fi

# Ensure the link directory exists.
if [[ ! -d "${AIRLOCK_V2_LINK_DIR}" ]]; then
    echo "[install_airlock_v2] Creating ${AIRLOCK_V2_LINK_DIR}..."
    mkdir -p "${AIRLOCK_V2_LINK_DIR}"
fi

ln -s "${AIRLOCK_V2_BIN}" "${AIRLOCK_V2_LINK}"
echo "[install_airlock_v2] Installed ${AIRLOCK_V2_LINK} -> ${AIRLOCK_V2_BIN}"

# 4) Verify the on-PATH binary works.
if ! command -v airlock-v2 >/dev/null 2>&1; then
    echo "[install_airlock_v2] WARNING: ${AIRLOCK_V2_LINK_DIR} is not on PATH for this shell." >&2
    echo "[install_airlock_v2] Add it: export PATH=\"${AIRLOCK_V2_LINK_DIR}:\${PATH}\"" >&2
fi

if command -v airlock-v2 >/dev/null 2>&1; then
    echo "[install_airlock_v2] OK: $(airlock-v2 --version) is on PATH"
fi

# 5) Register this repo (idempotent — register is a no-op if already registered).
if command -v airlock-v2 >/dev/null 2>&1; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if airlock-v2 register "${REPO_ROOT}" 2>/dev/null; then
        echo "[install_airlock_v2] Registered ${REPO_ROOT} with airlock-v2"
    fi
fi
#!/usr/bin/env bash
# install_pre_push_hook.sh — install the gated snapshot.sh as a git pre-push hook.
#
# Wires scripts/snapshot.sh into .git/hooks/pre-push so every `git push` runs
# the 6 release gates before any push is allowed. If a gate fails, the push
# is blocked with the snapshot.sh exit code.
#
# Idempotent: if .git/hooks/pre-push already points to this installer and is
# executable, this script exits 0 without changes.
#
# Environmental safety: if scripts/snapshot.sh is missing (e.g. partial clone,
# fresh checkout with submodules not initialized), the hook prints a warning
# and exits 0 — we never block a push purely because the local environment
# is incomplete. Gates are best-effort, push must always be reachable.
#
# Usage:
#   bash scripts/install_pre_push_hook.sh
#
# To run the gates manually without invoking airlock-v2 snapshot:
#   DRY_RUN=1 bash scripts/snapshot.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="${GIT_DIR:-${REPO_ROOT}/.git}/hooks"
HOOK_PATH="${HOOKS_DIR}/pre-push"

log() { printf '[install_pre_push_hook] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1) Locate or create the hooks directory.
# ---------------------------------------------------------------------------
if [[ ! -d "${HOOKS_DIR}" ]]; then
    log "Creating hooks directory: ${HOOKS_DIR}"
    mkdir -p "${HOOKS_DIR}"
fi

# ---------------------------------------------------------------------------
# 2) Build the hook payload.
# ---------------------------------------------------------------------------
# Self-contained: finds the repo root via `git rev-parse --show-toplevel`
# (works under worktrees too), then invokes scripts/snapshot.sh. If the
# snapshot script is missing, warn and allow the push — environment may be
# partial (no airlock, fresh submodule checkout, etc.). A missing gate runner
# must never brick a developer's push.
read -r -d '' HOOK_BODY <<'HOOK_EOF' || true
#!/usr/bin/env bash
# phenotype-omlx pre-push hook — gated by scripts/snapshot.sh.
# Installed by scripts/install_pre_push_hook.sh. Do not edit by hand;
# re-run the installer to regenerate.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SNAPSHOT_SCRIPT="${REPO_ROOT}/scripts/snapshot.sh"

if [[ ! -f "${SNAPSHOT_SCRIPT}" ]]; then
    printf '[pre-push] WARNING: %s not found — skipping gates (env may be partial)\n' \
        "${SNAPSHOT_SCRIPT}" >&2
    exit 0
fi

# Honour DRY_RUN so callers can pre-flight gates without invoking airlock.
# Local snapshot.sh reads DRY_RUN and skips the actual `airlock-v2 snapshot`.
export DRY_RUN="${DRY_RUN:-0}"

printf '[pre-push] running gated snapshot: bash %s (DRY_RUN=%s)\n' \
    "${SNAPSHOT_SCRIPT}" "${DRY_RUN}"
bash "${SNAPSHOT_SCRIPT}"
HOOK_EOF

# ---------------------------------------------------------------------------
# 3) Idempotency check — if the existing hook already matches our payload,
#    skip writing.
# ---------------------------------------------------------------------------
if [[ -f "${HOOK_PATH}" ]]; then
    if grep -q "phenotype-omlx pre-push hook" "${HOOK_PATH}" \
        && grep -q "install_pre_push_hook.sh" "${HOOK_PATH}"; then
        if [[ -x "${HOOK_PATH}" ]]; then
            log "Already installed and executable: ${HOOK_PATH}"
            exit 0
        fi
        log "Hook present but not executable — fixing permissions"
        chmod +x "${HOOK_PATH}"
        exit 0
    fi
    log "Existing hook at ${HOOK_PATH} is not ours — overwriting"
fi

# ---------------------------------------------------------------------------
# 4) Write and chmod the hook.
# ---------------------------------------------------------------------------
printf '%s\n' "${HOOK_BODY}" > "${HOOK_PATH}"
chmod +x "${HOOK_PATH}"

log "Installed: ${HOOK_PATH} (-> scripts/snapshot.sh)"
log "Manual run: DRY_RUN=1 bash scripts/snapshot.sh"
#!/usr/bin/env bash
# Justification: thin Bash glue around `git worktree add` for Phenotype layout.
# Prefer: repos/worktrees/phenotype-omlx/<branch> (not phenotype-omlx-tmp).
set -euo pipefail
BRANCH="${1:?usage: scripts/worktree_add.sh <branch>}"
MAIN="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
exec git -C "$MAIN" worktree add "../worktrees/phenotype-omlx/${BRANCH}" -b "${BRANCH}"

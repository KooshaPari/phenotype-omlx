#!/usr/bin/env bash
# helios_wrapper.sh — thin stock wrapper that delegates to the homebrew
# codex-cli binary. No custom agent loop; this is purely a passthrough so
# downstream phenotype-omlx tooling can invoke `codex` without baking a
# hardcoded path into every script.
#
# Rationale (per the previous session notes): the ForgeCode fork we want
# to use for vPU-aware agent operations is pending GitHub restore, so the
# safest fallback is the unmodified upstream codex-cli. We do NOT subclass,
# monkey-patch, or wrap with our own orchestration logic — that's an
# anti-pattern per AGENTS.md ("no custom agent loop").
#
# Usage:
#   ./scripts/helios_wrapper.sh <codex-args...>
#
# Environment overrides:
#   CODEX_BIN  — explicit path to the codex binary
#                (default: discovers via brew or falls back to the
#                 Caskroom path installed by `brew install --cask codex`)

set -euo pipefail

CODEX_BIN="${CODEX_BIN:-}"

if [[ -z "${CODEX_BIN}" ]]; then
  # Prefer the symlink if it resolves, otherwise dig into the Caskroom.
  if command -v codex >/dev/null 2>&1; then
    CODEX_BIN="$(command -v codex)"
  elif [[ -x "/opt/homebrew/Caskroom/codex"/*/codex-aarch64-apple-darwin ]]; then
    CODEX_BIN="$(ls -1 /opt/homebrew/Caskroom/codex/*/codex-aarch64-apple-darwin | head -n1)"
  fi
fi

if [[ -z "${CODEX_BIN}" || ! -x "${CODEX_BIN}" ]]; then
  echo "helios_wrapper: codex binary not found. Install with: brew install --cask codex" >&2
  exit 127
fi

exec "${CODEX_BIN}" "$@"

#!/usr/bin/env bash
# Print Langfuse MCP Basic auth material from apps/bench-cockpit/.env
# shellcheck disable=SC1091
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${LANGFUSE_ENV_FILE:-$ROOT/.env}"
TOKEN_ONLY=0
[[ "${1:-}" == "--token-only" ]] && TOKEN_ONLY=1

die() { echo "error: $*" >&2; exit 1; }

[[ -f "$ENV_FILE" ]] || die "missing $ENV_FILE"

# shellcheck disable=SC2046
eval "$(/usr/bin/grep -E '^(LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY|LANGFUSE_BASE_URL|LANGFUSE_HOST)=' "$ENV_FILE" | /usr/bin/sed 's/^/export /')"

[[ -n "${LANGFUSE_PUBLIC_KEY:-}" && -n "${LANGFUSE_SECRET_KEY:-}" ]] || die "LANGFUSE_PUBLIC_KEY/SECRET_KEY required"

BASE="${LANGFUSE_BASE_URL:-${LANGFUSE_HOST:-https://us.cloud.langfuse.com}}"
BASE="${BASE%/}"
TOKEN="$(printf '%s:%s' "$LANGFUSE_PUBLIC_KEY" "$LANGFUSE_SECRET_KEY" | /usr/bin/base64 | tr -d '\n')"

if [[ "$TOKEN_ONLY" -eq 1 ]]; then
  printf '%s\n' "$TOKEN"
  exit 0
fi

cat <<EOF
mcp_url=${BASE}/api/public/mcp
authorization=Basic ${TOKEN}
# Cursor / Claude: use Authorization header value above (includes "Basic ")
# Prefer Agent Skill when agents have shell: npx skills add langfuse/skills --skill langfuse
EOF

#!/usr/bin/env bash
# Emit Cursor MCP JSON for Langfuse (cloud or self-host from .env).
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HELPER="$ROOT/scripts/langfuse/mcp-auth-header.sh"

export MCP_URL="$(bash "$HELPER" | sed -n 's/^mcp_url=//p')"
export TOKEN="$(bash "$HELPER" --token-only)"

python3 -c '
import json, os
print(json.dumps({
  "mcp": {
    "servers": {
      "langfuse": {
        "url": os.environ["MCP_URL"],
        "headers": {"Authorization": "Basic " + os.environ["TOKEN"]},
      }
    }
  }
}, indent=2))
print("\n# Paste into Cursor → Settings → Tools & Integrations → Add Custom MCP")
print("# Prefer read-only tool allowlist for shared/prod projects.")
'

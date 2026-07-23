# Langfuse MCP, CLI, and Agent Skill

Bring Langfuse into terminals and coding agents (Cursor, Claude Code). Prefer the
**Agent Skill** when the agent can run bash; use **MCP** for IDE-native tools;
use **CLI** for scripts.

Project keys live in gitignored `apps/bench-cockpit/.env`
(`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`).

## Auth header

```bash
bash apps/bench-cockpit/scripts/langfuse/mcp-auth-header.sh
# prints: Basic <base64(pk:sk)>
# and the MCP URL for cloud or self-host
```

## Agent Skill (recommended for agents with shell)

```bash
npx skills add langfuse/skills --skill "langfuse"
```

Docs: https://langfuse.com/docs/docs — skill conditions agents on Langfuse
best practices (prompts, datasets, scores).

## MCP server

Endpoints:

| Deployment | URL |
|------------|-----|
| Cloud US (current cockpit default) | `https://us.cloud.langfuse.com/api/public/mcp` |
| Cloud EU | `https://cloud.langfuse.com/api/public/mcp` |
| Self-host | `${LANGFUSE_BASE_URL}/api/public/mcp` (e.g. `http://127.0.0.1:3000/api/public/mcp`) |

### Cursor

`bash apps/bench-cockpit/scripts/langfuse/print-cursor-mcp-snippet.sh` emits JSON
for **Settings → Tools & Integrations → Add Custom MCP**. Restrict to read-only
tools via your client allowlist if agents should not mutate production data.

### Claude Code

```bash
TOKEN="$(bash apps/bench-cockpit/scripts/langfuse/mcp-auth-header.sh --token-only)"
BASE="${LANGFUSE_BASE_URL:-https://us.cloud.langfuse.com}"
claude mcp add --transport http langfuse \
  "${BASE%/}/api/public/mcp" \
  --header "Authorization: Basic ${TOKEN}"
```

## CLI

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
# optional for self-host:
# export LANGFUSE_HOST="http://127.0.0.1:3000"

npx langfuse-cli api <resource> <action>
```

Same key pair as SDKs / cockpit BFF. Use for traces, prompts, datasets, scores.

## Cockpit pairing

| Goal | Command / UI |
|------|----------------|
| Sync hosted Minimax judges | Langfuse view → **Sync** or `setup_langfuse_judges.py` |
| Seed cells | **Seed traces + generations** |
| Offline scores | **Offline Minimax → scores** |
| Agent R/W | MCP / Skill / CLI above |

Cloud primary: `docs/guides/LANGFUSE_CLOUD.md`.
Org integrations: `docs/guides/LANGFUSE_ORG_INTEGRATIONS.md`.
Self-host overflow: `docs/guides/LANGFUSE_SELF_HOST.md`.
Alternatives decision: `docs/research/LANGFUSE_ALTERNATIVES.md`.

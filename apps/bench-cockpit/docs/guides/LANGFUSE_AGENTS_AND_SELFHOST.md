# Langfuse agents, MCP/CLI, and Cloud

Primary observability for bench-cockpit: **Langfuse Cloud Hobby**
(`https://us.cloud.langfuse.com`). Self-host is optional overflow — see
repo `docs/guides/LANGFUSE_SELF_HOST.md`.

## Why Langfuse (vs alternatives)

| Tool | Role |
|------|------|
| **Langfuse Cloud Hobby** | Primary: traces, scores, LLM-as-judge, dashboards, datasets, MCP/CLI |
| Langfuse self-host (MIT) | When Hobby caps or private-network needs appear |
| Phoenix (Arize, ELv2) | Strong #2 if you want OTel-native and accept ELv2 |
| Promptfoo | Add for CI/red-team gates — not a trace hub |
| Braintrust | Excellent eval UX; closed / enterprise hybrid |
| LangSmith | Leaving for cost + closed self-host |

**Verdict:** Stay on Langfuse Cloud. Add Promptfoo for agent CI if needed.
Do not run two full UIs.

## Cloud vs self-host

Cloud Hobby has unit caps, retention walls, seat limits, and API throttles.
Self-host removes those product gates at the cost of ops (PG/CH/MinIO/disk).

Recommended:

1. **Bench + agent runs → Cloud Hobby** (default).
2. Bootstrap: `python3 scripts/evals/setup_langfuse_cloud.py`
3. Self-host only when caps/fork force it; one-way export/import if sharing.
4. Minimax LLM connection (Cloud UI or existing upsert):
   - Provider `Minimax`, adapter `anthropic`
   - Base `https://api.minimax.io/anthropic`, model `Minimax-M3`

Org-wide matrix: `docs/guides/LANGFUSE_ORG_INTEGRATIONS.md`.

## Agent Skill

```bash
npx skills add langfuse/skills --skill "langfuse"
```

## CLI

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"

npx langfuse-cli api <resource> <action>
```

## MCP

```bash
# token = base64(publicKey:secretKey)
claude mcp add --transport http langfuse \
  https://us.cloud.langfuse.com/api/public/mcp \
  --header "Authorization: Basic {your-base64-token}"
```

Or: `bash scripts/langfuse/print-cursor-mcp-snippet.sh`

## Cockpit wiring

```bash
OBSERVABILITY_BACKEND=langfuse
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

python3 scripts/evals/setup_langfuse_cloud.py
python3 scripts/evals/setup_langfuse_judges.py      # hosted Minimax judges
# Historical V5 cells (no new stock_vs_ours trials) → traces → live scores
python3 scripts/evals/run_langfuse_evaluators.py all --limit 5
```

Dashboard: Cloud → **bench-cockpit-ops**. `all` fails loudly if Minimax keys are missing
(no silent `judge_score=0`).

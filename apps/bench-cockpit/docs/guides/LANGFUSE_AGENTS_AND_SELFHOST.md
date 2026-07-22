# Langfuse agents, MCP/CLI, and self-host

Primary observability for bench-cockpit. Prefer **self-host** as system of record;
cloud Hobby is optional for demos/collab only.

## Why Langfuse (vs alternatives)

| Tool | Role |
|------|------|
| **Langfuse (self-host MIT)** | Best primary: traces, scores, LLM-as-judge, playground, datasets, **Agent Skill + CLI + MCP** |
| Phoenix (Arize, ELv2) | Strong #2 if you want OTel-native and accept ELv2 |
| Promptfoo | Add for CI/red-team gates — not a trace hub |
| Braintrust | Excellent eval UX; closed / enterprise hybrid — not OSS-control fit |
| LangSmith | **Removed** (archived under `.archive/langsmith-20260722/`) |
| Helicone / Weave / Evidently / Ragas | Gateway, W&B, metrics sidecar, RAG metrics — complements only |

**Verdict:** Stay on Langfuse. Add Promptfoo for agent CI if needed. Do not run two full UIs.

## Cloud vs self-host (freemium)

Cloud Hobby has unit caps, retention walls, seat limits, and API throttles.
Self-host MIT has **no product gates** on those dimensions — same code path, richer for
bench volume. There is **no native bidirectional sync**.

Recommended:

1. **Bench + agent runs → self-host only** (Podman / Apple Container — never Docker).
2. Cloud Hobby → optional satellite (demos, guests).
3. Promote datasets/prompts **one-way** (export API → import) when sharing.
4. Mirror Minimax LLM connection on self-host:
   - Provider `Minimax`, adapter `anthropic`
   - Base `https://api.minimax.io/anthropic`, model `Minimax-M3`

## Agent Skill

```bash
npx skills add langfuse/skills --skill "langfuse"
```

Gives coding agents a playbook for instrumentation, traces, prompts, and evals.
Prefer Skill + CLI when the agent can run shell.

## CLI

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"  # or self-host URL

npx langfuse-cli api <resource> <action>
```

## MCP

Cloud (US):

```bash
# token = base64(publicKey:secretKey)
claude mcp add --transport http langfuse \
  https://us.cloud.langfuse.com/api/public/mcp \
  --header "Authorization: Basic {your-base64-token}"
```

Self-host (HTTPS preferred; localhost OK for local agents):

```bash
claude mcp add --transport http langfuse \
  http://127.0.0.1:3000/api/public/mcp \
  --header "Authorization: Basic {your-base64-token}"
```

Allowlist read-only tools if agents must not write.

## Cockpit wiring

```bash
OBSERVABILITY_BACKEND=langfuse
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # or http://127.0.0.1:3000

python3 scripts/evals/setup_langfuse_judges.py      # hosted Minimax judges
python3 scripts/evals/run_langfuse_evaluators.py sync|seed|judge
```

## Self-host deploy note

Use Langfuse’s container compose as **Podman Compose / Apple Container** manifests
(same images). Durable volumes under Phenotype paths — never `/tmp`.

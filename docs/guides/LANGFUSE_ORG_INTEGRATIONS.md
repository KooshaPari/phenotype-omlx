# Langfuse org-wide integrations

Phenotype standard: **one Langfuse Cloud project (or org)** as the observability
hub until Hobby limits force self-host. Repos should emit traces/scores into that
hub rather than inventing parallel UIs.

## Shared env contract

Every service that talks to Langfuse:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # US Hobby default
# optional aliases
# LANGFUSE_HOST=$LANGFUSE_BASE_URL
OBSERVABILITY_BACKEND=langfuse
```

Never commit keys. Prefer project `.env` / secret manager / keychain.

## Integration matrix

| Surface | Status in Phenotype | Recommended action |
|---------|---------------------|--------------------|
| bench-cockpit BFF + UI | **Done** | Keep Cloud as default |
| Hosted Minimax judges | **Done** (Cloud) | `setup_langfuse_judges.py` |
| Custom dashboards | **Done** (API bootstrap) | `setup_langfuse_cloud.py` |
| MCP / CLI / Agent Skill | **Documented** | `print-cursor-mcp-snippet.sh` |
| Python SDK (`langfuse`) | Cockpit scripts / seed | Use in any Python agent service |
| JS/TS SDK | Not org-standard yet | Add for helios / web agents |
| OpenTelemetry → Langfuse | **Not rolled out** | Prefer for Go/Rust services |
| Harbor / portage runs | **harbor-langfuse** (portage #478) + omlx `--langfuse` | Prefer over LangSmith |
| pheno-harness | No Langfuse refs found | Emit traces from harness runner |
| Slack / monitors / PostHog | **UI-only** | Configure once in Cloud console |
| Prompt management | Seed prompt `bench-judge-system` | Version prompts per product |
| Annotation queues | Hobby **1** queue | `bench-manual-review` |
| Blob export | Higher tier | Skip on Hobby |

## OpenTelemetry (services without Langfuse SDK)

Langfuse accepts OTLP. Pattern:

1. Instrument with OpenTelemetry (existing Phenotype OTel where present).
2. Export to Langfuse OTel endpoint for the Cloud region (see Langfuse docs:
   Integrations → OpenTelemetry).
3. Map `service.name` / `deployment.environment` so dashboards can filter.

Use this for Go/Rust/CLI tools before adding a second proprietary client.

## Cross-repo reuse

Candidate shared module (when a second repo adopts Langfuse):

- thin env loader + Basic-auth helper (today duplicated in cockpit Python scripts)
- target: `phenotype-shared` or a tiny `libs/observability-langfuse` package
- do **not** duplicate dashboard JSON — keep definitions in cockpit
  `setup_langfuse_cloud.py` until a second consumer appears

## Agent access

```bash
# Cursor MCP JSON
bash apps/bench-cockpit/scripts/langfuse/print-cursor-mcp-snippet.sh

# Skill
npx skills add langfuse/skills --skill "langfuse"
```

Allowlist read-only MCP tools for agents that must not mutate production prompts
or dashboards.

## Bootstrap order (new machine / new project)

1. Create Cloud project + keys → gitignored `.env`
2. UI: Minimax LLM connection (if not already present)
3. `python3 scripts/evals/setup_langfuse_cloud.py`
4. Seed / judge via cockpit or `run_langfuse_evaluators.py`
5. UI: Slack + 1–2 monitors
6. Wire Harbor/harness exporters when those paths leave LangSmith

## Caps watch

Hobby: ~50k units/mo, 30d retention, 2 users, 1 annotation queue, 2 monitors,
API rate limits. When any of these block work, revisit
[`LANGFUSE_SELF_HOST.md`](./LANGFUSE_SELF_HOST.md).

# Langfuse Cloud (primary)

**Primary observability for Phenotype / bench-cockpit:** Langfuse **Cloud Hobby**
(`https://us.cloud.langfuse.com`) until Hobby caps bite or there is a meaningful
self-host / fork product change.

Self-host remains available as a lab / overflow path — see
[`LANGFUSE_SELF_HOST.md`](./LANGFUSE_SELF_HOST.md). Do **not** treat local Apple
Container bring-up as the default day-to-day path.

## Why Cloud now

| Concern | Cloud Hobby | Self-host |
|---------|-------------|-----------|
| Setup time | minutes | hours (PG/CH/MinIO/VM disk) |
| Features | dashboards, judges, MCP, prompts, datasets | same code |
| Caps | 50k units/mo, 30d retention, 2 seats | infra only |
| Org sharing | shared project + MCP | extra ops |

## Cockpit wiring

```bash
# apps/bench-cockpit/.env (gitignored)
OBSERVABILITY_BACKEND=langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

# Minimax for hosted + offline judges
MINIMAX_API_KEY=...
MINIMAX_JUDGE_MODEL=MiniMax-M3
LANGFUSE_JUDGE_PROVIDER=Minimax
LANGFUSE_JUDGE_MODEL=Minimax-M3
```

One-shot bootstrap (score configs, datasets, prompt, annotation queue,
custom dashboard/widgets, then hosted judges):

```bash
cd apps/bench-cockpit
python3 scripts/evals/setup_langfuse_cloud.py
python3 scripts/evals/setup_langfuse_cloud.py --status
python3 scripts/evals/run_langfuse_evaluators.py seed --limit 40
```

UI: cockpit **Langfuse** view → Sync / Seed / Offline scores.
Dashboard: open Cloud → **Dashboards → bench-cockpit-ops**.

## Features we automate

| Feature | Script / path |
|---------|----------------|
| Score configs | `setup_langfuse_cloud.py` + `setup_langfuse_judges.py` |
| Datasets | `bench-cockpit-v5-cells` (+ harbor/harness names if present) |
| Hosted LLM-as-judge | Minimax connection + `setup_langfuse_judges.py` |
| Custom dashboard | `bench-cockpit-ops` + widgets via `/api/public/unstable/*` |
| Prompt library | `bench-judge-system` |
| Annotation queue | `bench-manual-review` (Hobby max **1**) |
| MCP / CLI / Skill | [`LANGFUSE_MCP_CLI.md`](./LANGFUSE_MCP_CLI.md) |

## UI-only integrations (do in Cloud console)

Hobby supports these; there is no stable project-key create API for most:

1. **Slack** — Settings → Integrations (ops channel for score/latency alerts)
2. **Monitors** — up to **2** on Hobby (e.g. p95 latency, mean correctness)
3. **Prompt webhooks / automations** — when production labels change
4. **PostHog / Mixpanel** — if product analytics is live org-wide
5. **Home dashboard** — set `bench-cockpit-ops` as project Home if desired

Blob-storage batch export needs org keys / higher tiers — skip on Hobby.

## Org-wide integrations

See [`LANGFUSE_ORG_INTEGRATIONS.md`](./LANGFUSE_ORG_INTEGRATIONS.md) for SDKs,
OpenTelemetry, Harbor/harness wiring, and shared env conventions.

## When to leave Cloud

Switch primary to self-host / fork when any of:

- sustained Hobby unit or retention wall
- need >2 seats or private network only
- meaningful Langfuse fork with Phenotype-specific features

Until then: keep Cloud keys in gitignored `.env`, rotate if pasted in chat,
and use self-host only for offline experiments.

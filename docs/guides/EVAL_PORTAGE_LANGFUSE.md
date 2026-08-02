# Eval ownership — Portage / Langfuse / omlx

> **Rule:** mature evals run through **Portage (Harbor)** + **required
> `harbor-langfuse`**. Langfuse is the **canonical, non-optional** observability
> backend. **LangSmith is removed** from the operator path — if Portage or
> Langfuse ingest is buggy, **fix Portage**; do not fall back to LangSmith.

## Layers

| Layer | Owns | Does not own |
|-------|------|----------------|
| `config/smoke_models.json` + `omlx_research.smoke_models` | Qwen3.5 model ids for smoke / FR | Long-running agent trials |
| **Portage** (`PORTAGE_ROOT`) | Harbor tasks, datasets, sandboxes, RL rollouts | Hardcoded MLX HF paths in Python scripts |
| **`harbor-langfuse` plugin** | Session=job, trace=trial, score=`reward` → Langfuse | Local-only EvaluationReport JSON |
| **pheno-harness** `bench/` | EvaluationReport contracts, RLVR verifiers | Harbor fork sources |
| **bench-cockpit** | Operator UI + Langfuse panel | Inventing new eval frameworks |

## Operator commands

```bash
export PORTAGE_ROOT=/path/to/portage/worktree   # required — no hardcoded default
export HARBOR_ENV=apple-container
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # or self-host
export PHENO_EXECUTION_WINDOW_ID=qwen35-20260802T0700Z-3090ti  # operator-issued, bounded window

# Harbor hello-world (oracle) — always attaches Langfuse
bash scripts/evals/run_via_harbor.sh

# OMLX Qwen3.5 policy Harbor task
bash scripts/evals/run_via_harbor.sh --policy

# NIAH via OpenAI-compatible omlx/MLX server (Qwen3.5 SSOT).
# Apple Container runs in a VM/network namespace: do not use host localhost.
# Bind MLX to 0.0.0.0 and use the host's LAN address (the runner auto-detects
# en0/en1 when OPENAI_BASE_URL is omitted).
export OPENAI_BASE_URL=http://$(ipconfig getifaddr en0):8766/v1
bash scripts/evals/run_via_harbor.sh --niah

# TurboQuant SSOT gate
bash scripts/evals/run_via_harbor.sh --turbo
```

`PHENO_EXECUTION_WINDOW_ID` is mandatory for every Harbor invocation. The runner
rejects missing or malformed IDs before checking Portage, starting Apple Container,
or invoking `uv run harbor`; this is an authorization marker, not proof that a
runtime workload was executed. Do not invent one or reuse a stale window.

`--langsmith` is **rejected** (exit 2). `--langfuse` is accepted for back-compat but redundant.

## Harbor tasks in this repo

| Task | Path | Needs |
|------|------|-------|
| Policy | `evals/harbor/tasks/omlx-qwen35-policy` | — |
| NIAH API smoke | `evals/harbor/tasks/omlx-niah-api-smoke` | `OPENAI_BASE_URL` |
| Turbo SSOT gate | `evals/harbor/tasks/omlx-turbo-ssot` | — |

## KPI / session mapping

See `config/langfuse_harbor_kpis.json`. Primary verifier score lands as Langfuse
score name `reward`; `sessionId` = Harbor `job_id`.

## Offline quality gates (no MLX)

```bash
python3 scripts/evals/contamination_scan.py
python3 scripts/evals/rlvr_af_smoke.py
```

## Deprecation

- Prefer Harbor JobConfig / tasks over growing `scripts/niah_*.py`.
- LangSmith panel / `harbor-langsmith` / `config/langsmith_harbor_kpis.json` are
  **legacy** — do not extend. Use Langfuse only.

## Self-host notes

Before Harbor runs, the operator verifies `container system status` and starts
the service with `container system start` when needed. If it still does not
reach `running`, inspect `container system logs`. The public Apple Container
runtime intentionally isolates guest localhost; use a routable host LAN IP or
another host-reachable endpoint for the OpenAI-compatible server.

Default UI: `http://127.0.0.1:3000`. Host Homebrew Postgres via Apple Container gateway:
`DATABASE_URL=postgresql://langfuse:langfuse@192.168.65.1:5432/langfuse` (do not reuse stale `192.168.64.*` IPs).

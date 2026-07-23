# Eval ownership — Portage / Langfuse / omlx

> **Rule:** mature evals run through **portage (Harbor)** + **harbor-langfuse**.
> LangSmith (`harbor-langsmith`) is optional legacy. Ad-hoc `scripts/*.py` are
> thin adapters — not the operator SSOT.

See also: [`LANGFUSE_CLOUD.md`](./LANGFUSE_CLOUD.md),
[`LANGFUSE_ORG_INTEGRATIONS.md`](./LANGFUSE_ORG_INTEGRATIONS.md),
KPI SSOT: `config/langfuse_harbor_kpis.json`.

## Layers

| Layer | Owns | Does not own |
|-------|------|----------------|
| `config/smoke_models.json` + `omlx_research.smoke_models` | Qwen3.5 model ids for smoke / FR | Long-running agent trials |
| **portage** (`PORTAGE_ROOT`) | Harbor tasks, datasets, sandboxes, RL rollouts | Hardcoded MLX HF paths in Python scripts |
| **harbor-langfuse** plugin | Job→session, trial→trace, reward→score on Langfuse Cloud | Local-only EvaluationReport JSON |
| **harbor-langsmith** plugin | Legacy LangSmith mirror | New Phenotype eval work |
| **pheno-harness** `bench/` | EvaluationReport contracts, RLVR verifiers | Harbor fork sources |
| **bench-cockpit** | Operator UI + `POST /api/eval/run` → Portage bridge | Inventing new eval frameworks |

## Operator commands

```bash
export PORTAGE_ROOT=/path/to/portage/worktree   # required — no hardcoded default
export HARBOR_ENV=apple-container
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

# Harbor hello-world (oracle)
bash scripts/evals/run_via_harbor.sh

# + Langfuse plugin (primary)
bash scripts/evals/run_via_harbor.sh --langfuse

# Cockpit smoke helper
bash apps/bench-cockpit/scripts/evals/harbor_langfuse_smoke.sh

# Legacy LangSmith mirror (optional)
export LANGSMITH_API_KEY=...
bash scripts/evals/run_via_harbor.sh --langsmith

# OMLX Qwen3.5 policy Harbor task
bash scripts/evals/run_via_harbor.sh --policy --langfuse

# NIAH via OpenAI-compatible omlx/MLX server (Qwen3.5 SSOT)
export OPENAI_BASE_URL=http://127.0.0.1:8766/v1
bash scripts/evals/run_via_harbor.sh --niah --langfuse

# TurboQuant SSOT gate
bash scripts/evals/run_via_harbor.sh --turbo --langfuse
```

Cockpit BFF: `POST /api/eval/run` with `{"mode":"hello_world"}` auto-attaches
`--plugin langfuse` when `OBSERVABILITY_BACKEND=langfuse`. Explicit:
`{"plugin_langfuse":true}`.

## Harbor tasks in this repo

| Task | Path | Needs |
|------|------|-------|
| Policy | `evals/harbor/tasks/omlx-qwen35-policy` | — |
| NIAH API smoke | `evals/harbor/tasks/omlx-niah-api-smoke` | `OPENAI_BASE_URL` |
| Turbo SSOT gate | `evals/harbor/tasks/omlx-turbo-ssot` | — |

## Offline quality gates (no MLX)

```bash
python3 scripts/evals/contamination_scan.py
python3 scripts/evals/rlvr_af_smoke.py
```

## Deprecation

- Prefer Langfuse Cloud until Hobby caps; self-host overflow only.
- Do not add new LangSmith-only operator paths.
- Legacy guide name `EVAL_PORTAGE_LANGSMITH.md` redirects here conceptually —
  this file is the SSOT.

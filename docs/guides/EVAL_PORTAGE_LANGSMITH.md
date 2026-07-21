# Eval ownership — Portage / LangSmith / omlx

> **Rule:** mature evals run through **portage-TEMP (Harbor)** + optional
> **harbor-langsmith**. Ad-hoc `scripts/*.py` are thin adapters or legacy
> instruments — not the operator SSOT.

## Layers

| Layer | Owns | Does not own |
|-------|------|----------------|
| `config/smoke_models.json` + `omlx_research.smoke_models` | Qwen3.5 model ids for smoke / FR | Long-running agent trials |
| **portage-TEMP** (`PORTAGE_ROOT`) | Harbor tasks, datasets, sandboxes, RL rollouts | Hardcoded MLX HF paths in Python scripts |
| **harbor-langsmith** plugin | Experiment/dataset sync to LangSmith | Local-only EvaluationReport JSON |
| **pheno-harness** `bench/` | EvaluationReport contracts, RLVR verifiers | Harbor fork sources |
| **bench-cockpit** | Operator UI + `POST /api/eval/run` → Portage bridge | Inventing new eval frameworks |

## Operator commands

```bash
export PORTAGE_ROOT=/path/to/portage/worktree   # required — no hardcoded default
export HARBOR_ENV=apple-container

# Harbor hello-world (oracle)
bash scripts/evals/run_via_harbor.sh

# + LangSmith plugin
export LANGSMITH_API_KEY=...
bash scripts/evals/run_via_harbor.sh --langsmith

# OMLX Qwen3.5 policy Harbor task
bash scripts/evals/run_via_harbor.sh --policy

# NIAH via OpenAI-compatible omlx/MLX server (Qwen3.5 SSOT)
export OPENAI_BASE_URL=http://127.0.0.1:8765/v1   # do not steal user's :8765 without asking
bash scripts/evals/run_via_harbor.sh --niah
# Host dry-run (no Harbor):
#   OPENAI_BASE_URL=... PYTHONPATH=python python3 scripts/evals/niah_openai_smoke.py

# TurboQuant SSOT gate (Metal TurboQuant+ stays host-side ready check 12)
bash scripts/evals/run_via_harbor.sh --turbo
```

JobConfig templates (documentation / future `harbor run -c`):  
`evals/harbor/jobs/niah-qwen35.yaml`, `evals/harbor/jobs/turboquant-ssot.yaml`.

## Harbor tasks in this repo

| Task | Path | Needs |
|------|------|-------|
| Policy | `evals/harbor/tasks/omlx-qwen35-policy` | — |
| NIAH API smoke | `evals/harbor/tasks/omlx-niah-api-smoke` | `OPENAI_BASE_URL` |
| Turbo SSOT gate | `evals/harbor/tasks/omlx-turbo-ssot` | — |

Resolve smoke model without running Harbor:

```bash
PYTHONPATH=python python3 -m omlx_research.smoke_models readiness
PYTHONPATH=python python3 -m omlx_research.smoke_models niah
```

## Qwen2.5 quarantine

- Defaults **must** be Qwen3.5 (FR-5 / org directive).
- Legacy id only with `OMLX_ALLOW_LEGACY_QWEN25=1` (local debug, never FR).
- Pre-retarget baselines: `.archive/qwen25-baselines/`.

## Deprecation

Prefer Harbor JobConfig / tasks over growing `scripts/niah_*.py`,
`perf_turboquant.py`, and dispatch stubs. Those scripts now read
`smoke_models`. Live NIAH for operators should use
`scripts/evals/run_via_harbor.sh --niah` (OpenAI-compatible endpoint).
Full Metal TurboQuant+ remains host `phenotype_omlx_ready.py` check 12;
Harbor `--turbo` only gates SSOT policy until a Metal-capable Harbor
environment exists.

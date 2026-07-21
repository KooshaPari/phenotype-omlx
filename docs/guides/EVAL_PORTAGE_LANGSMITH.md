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
```

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
`smoke_models` and should eventually become Harbor tasks that call into
omlx via OpenAI-compatible endpoints (`OPENAI_BASE_URL`).

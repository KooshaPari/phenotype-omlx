# bench-cockpit

LLM evals dashboard (Quality / Performance / RLVR / throughput) for MLX /
TurboQuant model runs. Listens on `:8090`.

**Active home:** `KooshaPari/phenotype-omlx-tmp` → `apps/bench-cockpit`
(the working stand-in while the deleted `phenotype-omlx` GitHub name stays
open for GH Support restore).

**Platform context:** eventual merge target is a unified inference /
AI·ML app platform that absorbs omlx + hwLedger (+ related). Cockpit
stays under omlx-tmp until that merge lands; hwLedger is the hardware /
capacity / fleet plane of the same platform.

**Not homes:** RepoLedger (separate project). Do not author here.

## Dev

Always use **bun** (never npm/yarn/pnpm):

# Prefer V5 EvaluationReport when present; override with BENCH_DATA.
# Clean smoke (no vacuous lint ERROR): BENCH_DATA=fixtures/smoke_results.json …
# Lint detector demo: BENCH_DATA=fixtures/smoke_lint_demo.json …
# Overview shows one Calibration chip (not ERROR banner spam); full lints on Calib view.

```bash
# one-shot (prefers V5 EvaluationReport when present)
bash scripts/start-dev.sh
# override data: BENCH_DATA=/path/to/results.json bash scripts/start-dev.sh
# clean smoke (no vacuous lint ERROR): BENCH_DATA=fixtures/smoke_results.json …
# lint detector demo: BENCH_DATA=fixtures/smoke_lint_demo.json …
```

**Suites view:** each suite is an expandable row; expand to per-task stock/ours
metrics (click a task to open the cell drawer). Overview suite cards jump here
and auto-expand the suite.

```bash
# or manual:
cd server && go run . \
  -data /Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours/run-v5-qwen35-08b-contract.json \
  -port 8090
bun install && bun run build && # serve via Go -dist ../dist
```

The Go server auto-detects pheno-harness **EvaluationReport v0.1**
(`contract_version` + `suites`) and flattens it to cockpit `{summary,cells}`.
Combined V5 contracts with duplicated suite blocks map first pass → `stock`,
second → `ours`. All-synthetic reports raise a `synthetic_100pct` lint error.

Package manager is pinned via `"packageManager": "bun@…"` in `package.json`.

## Observability (Langfuse primary)

**Recommendation:** use **Langfuse** for tracing + scores — MIT OSS, self-hostable with
full feature control and no SaaS meter. LangSmith remains optional/legacy.

```bash
# .env (gitignored)
OBSERVABILITY_BACKEND=langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # or self-host URL
```

- **Langfuse** view: seed V5 cells as traces + run Minimax judges onto Langfuse scores.
- CLI: `python3 scripts/evals/run_langfuse_evaluators.py status|seed|judge --limit 40`
- Self-host later via Podman / Apple Container (never Docker) for unlimited local control.

Minimax coding-plan key: `MINIMAX_API_KEY` or keychain `minimax-coding-plan`.

## LangSmith (optional legacy)

Put `LANGSMITH_API_KEY` in gitignored `.env` only if still needed.

### Hosted Minimax (OpenAI-compat)

1. **UI once:** Settings → Provider secrets (`MINIMAX_API_KEY`) → Model configurations
   (OpenAI Compatible Endpoint, base `https://api.minimax.io/v1`, model `MiniMax-M3`,
   temperature 0). Enable under Feature Access → Evaluators.
2. **API:** push Hub StructuredPrompts + register LLM evaluators + attach project run rules:

```bash
uv venv --python python3.12 .venv-evals
uv pip install --python .venv-evals/bin/python langsmith langchain-core
.venv-evals/bin/python scripts/evals/setup_hosted_judges.py
```

### Offline

- CLI: `python3 scripts/evals/run_evaluators.py sync|run|all --limit 20`
- Harbor smoke: `bash scripts/evals/harbor_langsmith_smoke.sh` (`-e apple-container`).

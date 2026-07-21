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

```bash
# one-shot (prefers V5 EvaluationReport when present)
bash scripts/start-dev.sh
# override data: BENCH_DATA=/path/to/results.json bash scripts/start-dev.sh

# or manual:
cd server && go run . \
  -data /Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours/run-v5-qwen35-08b-contract.json \
  -port 8090
bun install && bun run dev
```

The Go server auto-detects pheno-harness **EvaluationReport v0.1**
(`contract_version` + `suites`) and flattens it to cockpit `{summary,cells}`.
Combined V5 contracts with duplicated suite blocks map first pass → `stock`,
second → `ours`. All-synthetic reports raise a `synthetic_100pct` lint error.

Package manager is pinned via `"packageManager": "bun@…"` in `package.json`.

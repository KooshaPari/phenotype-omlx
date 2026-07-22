# bench-cockpit

LLM evals dashboard (Quality / Performance / RLVR / throughput) for MLX /
TurboQuant model runs. Listens on `:8090`.

**Active home:** `KooshaPari/phenotype-omlx` → `apps/bench-cockpit`
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
  -data /Users/kooshapari/CodeProjects/Phenotype/repos/pheno-harness/bench/results/stock-vs-ours/run-v5-qwen35-08b-contract.json \
  -port 8090
bun install && bun run build && # serve via Go -dist ../dist
```

The Go server auto-detects pheno-harness **EvaluationReport v0.1**
(`contract_version` + `suites`) and flattens it to cockpit `{summary,cells}`.
Combined V5 contracts with duplicated suite blocks map first pass → `stock`,
second → `ours`. All-synthetic reports raise a `synthetic_100pct` lint error.

Package manager is pinned via `"packageManager": "bun@…"` in `package.json`.

## Suite coverage (stock + experiment arms)

Default load: V5 `stock`/`ours` (10 suites) **plus** `minimax-m3-full/matrix.json`
as experiment arm `minimax-m3` (HLE, PinchBench, vending-bench, …).

```bash
# optional overrides
BENCH_DATA=/path/to/run-v5.json
BENCH_EXTRA_DATA=/path/to/minimax-m3-full/matrix.json
bash scripts/start-dev.sh
```

Overview → **Suite coverage** table shows paired stock/ours vs partial/missing
(catalog includes `ycbench` as unimplemented gap). Full stock+ours for every suite
still requires extending `pheno-harness` `stock_vs_ours.SUITES` and re-running.

Agents / MCP / self-host: `docs/guides/LANGFUSE_AGENTS_AND_SELFHOST.md`.

## Observability (Langfuse — required)

**Langfuse** is the canonical observability backend for tracing, playground, and
hosted LLM-as-judge — MIT OSS, self-hostable, full feature control. LangSmith
has been removed from the operator path; if Portage or Langfuse ingest is buggy,
fix Portage — there is no LangSmith fallback.

```bash
# .env (gitignored)
OBSERVABILITY_BACKEND=langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # or self-host URL
MINIMAX_API_KEY=...                               # offline judges + coding-plan
MINIMAX_JUDGE_MODEL=MiniMax-M3
```

### Hosted Minimax judges (preferred)

1. **UI once:** Langfuse → Settings → LLM Connections → custom provider:
   - Provider name: `Minimax`
   - Adapter: `anthropic`
   - Base URL: `https://api.minimax.io/anthropic`
   - Custom model: `Minimax-M3`
   - API key: MiniMax coding-plan key
2. **Sync evaluators + live observation rules:**

```bash
python3 scripts/evals/setup_langfuse_judges.py
# or: python3 scripts/evals/run_langfuse_evaluators.py sync
```

Creates project evaluators `bench-correctness` / `bench-hallucination` /
`bench-code-checker` (provider `Minimax`, model `Minimax-M3`) plus observation
rules. Seed then emits `generation-create` so those rules can score cells.

Playground and Evaluators UI use the same LLM connection (provider bills usage).

### Offline Minimax → Langfuse scores

Fallback when hosted preflight is flaky:

```bash
python3 scripts/evals/run_langfuse_evaluators.py seed --limit 40
python3 scripts/evals/run_langfuse_evaluators.py judge --limit 20
```

- **Langfuse** view: Sync hosted judges · Seed traces+generations · Offline scores.

### Self-host (preferred over freemium Cloud Hobby)

Unlimited units/retention vs Hobby (50k / 30d / 2 users). Same product images.

```bash
bash scripts/langfuse/self-host.sh init
bash scripts/langfuse/self-host.sh up     # Apple Container or Podman — never Docker
# then LANGFUSE_BASE_URL=http://127.0.0.1:3000 in .env
```

Guide: `../../docs/guides/LANGFUSE_SELF_HOST.md` (dual-write / migrate strategies).

### MCP · CLI · Agent Skill

```bash
npx skills add langfuse/skills --skill "langfuse"          # preferred for agents
bash scripts/langfuse/mcp-auth-header.sh                   # Basic token + MCP URL
bash scripts/langfuse/print-cursor-mcp-snippet.sh          # Cursor MCP JSON
npx langfuse-cli api <resource> <action>                   # full API from shell
```

Guide: `../../docs/guides/LANGFUSE_MCP_CLI.md`.

### Why Langfuse (not Phoenix / Braintrust / …)

Decision record: `../../docs/research/LANGFUSE_ALTERNATIVES.md` — Langfuse stays
primary; Phoenix/Braintrust only as optional side tools.

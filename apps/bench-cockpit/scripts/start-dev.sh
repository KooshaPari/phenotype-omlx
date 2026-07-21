#!/bin/bash
# Durable local start (macOS). Always bun for JS; Go for API.
# shellcheck disable=SC1091
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_DIR="${ROOT}/.run"
mkdir -p "$RUN_DIR"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

# Prefer V5 EvaluationReport when present; override with BENCH_DATA.
DEFAULT_V5="/Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours/run-v5-qwen35-08b-contract.json"
DATA_PATH="${BENCH_DATA:-}"
if [[ -z "$DATA_PATH" ]]; then
  if [[ -f "$DEFAULT_V5" ]]; then
    DATA_PATH="$DEFAULT_V5"
  else
    DATA_PATH="$ROOT/fixtures/smoke_results.json"
  fi
fi

(cd server && go build -o "$RUN_DIR/bench-cockpit-server" .)
[[ -d dist ]] || bun run build

# Stop prior instances of THIS binary only (do not kill unrelated :8090).
if [[ -f "$RUN_DIR/server.pid" ]]; then
  kill "$(cat "$RUN_DIR/server.pid")" 2>/dev/null || true
fi
if [[ -f "$RUN_DIR/vite.pid" ]]; then
  kill "$(cat "$RUN_DIR/vite.pid")" 2>/dev/null || true
fi
sleep 0.3

nohup "$RUN_DIR/bench-cockpit-server" \
  -data "$DATA_PATH" \
  -dist "$ROOT/dist" \
  -port "${BENCH_PORT:-8090}" \
  >"$RUN_DIR/server.log" 2>&1 &
echo $! >"$RUN_DIR/server.pid"

nohup "$ROOT/node_modules/.bin/vite" --host 127.0.0.1 --port "${VITE_PORT:-5173}" \
  >"$RUN_DIR/vite.log" 2>&1 &
echo $! >"$RUN_DIR/vite.pid"

sleep 1
PORT="${BENCH_PORT:-8090}"
echo "data: $DATA_PATH"
echo "Go:   http://127.0.0.1:${PORT}/"
echo "Vite: http://127.0.0.1:${VITE_PORT:-5173}/"
curl -fsS "http://127.0.0.1:${PORT}/api/health" || echo "health DOWN"
curl -fsS "http://127.0.0.1:${PORT}/api/state" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("cells", d.get("data",{}).get("summary",{}).get("meta",{}).get("n_cells"), "warnings", len(d.get("warnings") or []))' || echo "state DOWN"

#!/bin/bash
# Durable local start (macOS). Always bun for JS; Go for API.
# shellcheck disable=SC1091
#
# Data sources:
#   Real V5 (default):  bash scripts/start-dev.sh
#   Clean smoke:        BENCH_DATA=fixtures/smoke_results.json bash scripts/start-dev.sh
#   Lint demo:          BENCH_DATA=fixtures/smoke_lint_demo.json bash scripts/start-dev.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_DIR="${ROOT}/.run"
mkdir -p "$RUN_DIR"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

# Prefer V5 EvaluationReport when present; override with BENCH_DATA.
# Native JSON keeps richer per-cell fields (tok/s, traces); contract is thinner.
DEFAULT_V5_NATIVE="/Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours/run-v5-qwen35-08b.json"
DEFAULT_V5_CONTRACT="/Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours/run-v5-qwen35-08b-contract.json"
DATA_PATH="${BENCH_DATA:-}"
if [[ -z "$DATA_PATH" ]]; then
  if [[ -f "$DEFAULT_V5_NATIVE" ]]; then
    DATA_PATH="$DEFAULT_V5_NATIVE"
  elif [[ -f "$DEFAULT_V5_CONTRACT" ]]; then
    DATA_PATH="$DEFAULT_V5_CONTRACT"
  else
    DATA_PATH="$ROOT/fixtures/smoke_results.json"
  fi
elif [[ "$DATA_PATH" != /* ]]; then
  DATA_PATH="$ROOT/$DATA_PATH"
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

# Prefer a new session so the Go process survives shell exit (macOS).
python3 - "$RUN_DIR/bench-cockpit-server" "$DATA_PATH" "$ROOT/dist" "${BENCH_PORT:-8090}" "$RUN_DIR" <<'PY'
import os, subprocess, sys
bin, data, dist, port, run_dir = sys.argv[1:6]
log = open(os.path.join(run_dir, "server.log"), "a")
p = subprocess.Popen(
    [bin, "-data", data, "-dist", dist, "-port", port],
    stdout=log, stderr=log, start_new_session=True,
)
open(os.path.join(run_dir, "server.pid"), "w").write(str(p.pid))
print(p.pid)
PY

if [[ -x "$ROOT/node_modules/.bin/vite" ]]; then
  nohup "$ROOT/node_modules/.bin/vite" --host 127.0.0.1 --port "${VITE_PORT:-5173}" \
    >"$RUN_DIR/vite.log" 2>&1 &
  echo $! >"$RUN_DIR/vite.pid"
fi

sleep 1
PORT="${BENCH_PORT:-8090}"
echo "data: $DATA_PATH"
echo "Go:   http://127.0.0.1:${PORT}/"
echo "Vite: http://127.0.0.1:${VITE_PORT:-5173}/"
for _ in 1 2 3 4 5 6 7 8; do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then break; fi
  sleep 0.4
done
curl -fsS "http://127.0.0.1:${PORT}/api/health" || echo "health DOWN"
curl -fsS "http://127.0.0.1:${PORT}/api/state" | python3 -c '
import json,sys
d=json.load(sys.stdin)
data=d.get("data") or {}
cells=data.get("cells")
if cells is None and isinstance(data.get("data"), dict):
  cells=data["data"].get("cells")
print("cells", len(cells or []), "warnings", len(d.get("warnings") or []))
' || echo "state DOWN"

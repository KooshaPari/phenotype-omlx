#!/bin/bash
# Bootstrap LangSmith project + dataset + seeded runs for bench-cockpit.
# Requires LANGSMITH_API_KEY in .env (gitignored).
# shellcheck disable=SC1091
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ -z "${LANGSMITH_API_KEY:-}" ]]; then
  echo "LANGSMITH_API_KEY missing — write apps/bench-cockpit/.env first" >&2
  exit 1
fi
PORT="${BENCH_PORT:-8090}"
if ! curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
  echo "bench-cockpit not healthy on :${PORT} — run: bash scripts/start-dev.sh" >&2
  exit 1
fi
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/langsmith/setup" \
  -H 'Content-Type: application/json' \
  -d "{\"max_cells\":${LANGSMITH_MAX_CELLS:-40},\"seed_runs\":true}" \
  | python3 -m json.tool
echo
curl -fsS "http://127.0.0.1:${PORT}/api/langsmith/status" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("enabled", d.get("enabled"))
print("sessions", len(d.get("sessions") or []))
print("datasets", len(d.get("datasets") or []))
for s in (d.get("sessions") or [])[:8]:
  print(" -", s.get("name"), s.get("id"))
'

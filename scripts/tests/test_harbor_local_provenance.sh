#!/usr/bin/env bash
# Shell contract: local runner contains no plugin and exports no remote telemetry.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/portage" "$TMP/bin"

cat > "$TMP/bin/uv" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" > "$HARBOR_COMMAND_LOG"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then OUT="$2"; shift 2; continue; fi
  shift
done
mkdir -p "$OUT/trial"
cat > "$OUT/result.json" <<'JSON'
{"id":"local-job","started_at":"2026-07-26T00:00:00Z","finished_at":"2026-07-26T00:01:00Z","stats":{"evals":{"oracle__omlx-niah-api-smoke":{"n_trials":1,"metrics":[{"mean":1.0}]}}}}
JSON
cat > "$OUT/trial/result.json" <<'JSON'
{"id":"trial","trial_name":"trial","task_name":"omlx-niah-api-smoke","agent_info":{"name":"oracle","version":"1"},"agent_result":{"n_input_tokens":1,"n_output_tokens":1},"verifier_result":{"rewards":{"reward":1.0}},"started_at":"2026-07-26T00:00:00Z","finished_at":"2026-07-26T00:01:00Z","config":{"job_id":"local-job"}}
JSON
EOF
chmod +x "$TMP/bin/uv"

PATH="$TMP/bin:$PATH" HARBOR_PYTHON_BIN="$(command -v python3)" HARBOR_UV_BIN="$TMP/bin/uv" PORTAGE_ROOT="$TMP/portage" HARBOR_LOCAL_OUT="$TMP/out" \
  HARBOR_COMMAND_LOG="$TMP/command" OMLX_READY_MODEL="Qwen3.5-0.8B" \
  OPENAI_BASE_URL="http://127.0.0.1:8766/v1" \
  bash "$ROOT/scripts/evals/run_via_harbor_local.sh" --niah >/dev/null

grep -q -- 'harbor run' "$TMP/command"
grep -q -- ':8766/v1' "$TMP/command"
! grep -qi -- 'plugin\|langfuse\|trace\|session' "$TMP/command"
python3 - "$TMP/out/evaluation_report.local.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["telemetry"] == {"mode": "local_only", "remote_exported": False}
assert "langfuse" not in json.dumps(d).lower()
PY

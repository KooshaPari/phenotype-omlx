#!/usr/bin/env bash
set -euo pipefail
test -f /app/niah_answer.txt
grep -q '42-alpha' /app/niah_answer.txt
test -f /app/niah_result.json
python3 - <<'PY'
import json
d=json.load(open("/app/niah_result.json"))
assert d.get("exact_match") is True, d
assert "qwen3.5" in d.get("model","").lower(), d.get("model")
print("niah ok", d["model"])
PY

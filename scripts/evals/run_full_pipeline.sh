#!/usr/bin/env bash
# Full pipeline: Harbor → cockpit → EvalReport contract validation.
#
# Runs harbor_to_cockpit on the latest Harbor run, outputs to
# bench-cockpit fixtures, and prints a summary.
#
# Usage:
#   bash scripts/evals/run_full_pipeline.sh [--run-dir <path>]
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT/evals/harbor${PYTHONPATH:+:$PYTHONPATH}"

# ── resolve latest run ──
RUN_DIR=""
for arg in "$@"; do
	case "$arg" in
	--run-dir)
		shift_next=true
		;;
	*)
		if [[ "${shift_next:-}" == "true" ]]; then
			RUN_DIR="$arg"
			shift_next=false
		fi
		;;
	esac
done

if [[ -z "$RUN_DIR" ]]; then
	RUNS_BASE="$ROOT/.runs/harbor-eval-judge-resume"
	if [[ -d "$RUNS_BASE" ]]; then
		RUN_DIR="$(ls -1d "$RUNS_BASE"/*/ 2>/dev/null | tail -1 | sed 's:/$::')"
	fi
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
	echo "ERROR: No Harbor run directory found. Use --run-dir <path>." >&2
	exit 1
fi

RESULT_JSON="$RUN_DIR/result.json"
if [[ ! -f "$RESULT_JSON" ]]; then
	echo "ERROR: No result.json in $RUN_DIR" >&2
	exit 1
fi

OUT="$ROOT/apps/bench-cockpit/fixtures/harbor_oracle_results.json"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Full Pipeline: Harbor → cockpit → EvalReport"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Run dir:  $RUN_DIR"
echo "  Output:   $OUT"
echo ""

# ── Stage 1: Harbor → cockpit ──
echo "Stage 1: Harbor → cockpit cells"
python3 "$ROOT/scripts/evals/harbor_to_cockpit.py" "$RUN_DIR" -o "$OUT"
CELL_COUNT=$(python3 -c "import json; print(len(json.load(open('$OUT'))['cells']))")
echo "  → $CELL_COUNT cell(s) written"
echo ""

# ── Stage 2: Validate cockpit output ──
echo "Stage 2: Validate cockpit schema"
python3 -c "
import json, sys
d = json.load(open('$OUT'))
cells = d['cells']
meta = d['summary']['meta']
required = {'suite','task_id','difficulty','variant','ok','wall_clock_s','pass_at_1','metadata'}
for c in cells:
    missing = required - set(c.keys())
    if missing:
        print(f'  FAIL: cell {c.get(\"task_id\",\"?\")} missing {missing}', file=sys.stderr)
        sys.exit(1)
print(f'  → {len(cells)} cells pass schema check')
print(f'  Model: {meta[\"model\"]}')
print(f'  Variants: {meta[\"variants\"]}')
print(f'  n_cells: {meta[\"n_cells\"]}')
"
echo ""

# ── Stage 3: EvalReport contract ──
echo "Stage 3: EvalReport v1.0 contract validation"
python3 -c "
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path('$ROOT/evals/harbor')))
from interchange.contract import EvalReport
from interchange.validator import validate

cockpit = json.load(open('$OUT'))
cells = cockpit['cells']
meta = cockpit['summary']['meta']

# Group by suite
groups = {}
for c in cells:
    groups.setdefault(c['suite'], []).append(c)

suites = []
all_task_ids = []
for sn, sc in sorted(groups.items()):
    n = len(sc)
    passed = sum(1 for c in sc if c.get('ok'))
    pat = round(sum(c.get('pass_at_1', 0) for c in sc) / n, 4)
    tids = [c.get('task_id', '') for c in sc]
    all_task_ids.extend(tids)
    suites.append({'suite': sn, 'n': n, 'passed': passed, 'pass_at_1': pat, 'evidence_label': 'reported', 'task_ids': tids})

tc = len(cells)
tp = sum(1 for c in cells if c.get('ok'))

doc = {
    'contract_version': '1.0',
    'artifact_kind': 'EvaluationReport',
    'producer': {'name': 'harbor_to_cockpit', 'version': '1.0.0', 'commit_sha': 'pipeline-run'},
    'run': {'run_id': f'pipeline-{meta.get(\"model\",\"unknown\")}', 'started_at': '2026-07-22T22:39:39Z', 'model': meta.get('model','unknown'), 'variant': 'stock', 'judge_mode': 'deterministic'},
    'suites': suites,
    'totals': {'cells': tc, 'passed': tp, 'pass_at_1': round(tp/tc, 4) if tc else 0},
    'hash_chain': {'top_level_sha256': '', 'task_ids_sorted_sha256': hashlib.sha256(chr(10).join(sorted(all_task_ids)).encode()).hexdigest()},
}
payload = {k: v for k, v in doc.items() if k != 'hash_chain'}
doc['hash_chain']['top_level_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':')).encode()).hexdigest()

report = EvalReport.model_validate(doc)
result = validate(report, doc)

if not result.valid:
    print(f'  FAIL: {result.errors}', file=sys.stderr)
    sys.exit(1)

print(f'  → EvalReport v1.0 valid')
print(f'  Suites: {len(report.suites)}')
print(f'  pass_at_1: {report.totals.pass_at_1}')
print(f'  Warnings: {len(result.warnings)}')
"
echo ""

# ── Summary ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Pipeline complete. Fixture: $OUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

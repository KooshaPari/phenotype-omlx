#!/usr/bin/env bash
# Pull Mac V5 EvaluationReport into cockpit data/ when Tailscale Mac is reachable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/apps/bench-cockpit/data"
mkdir -p "$OUT"
SSH=(ssh -F "${HOME}/.ssh/config.pheno" -o ConnectTimeout=15 kooshas-laptop)
SCP=(scp -F "${HOME}/.ssh/config.pheno" -o ConnectTimeout=15)
REMOTE_DIR='~/CodeProjects/Phenotype/pheno-harness/bench/results/stock-vs-ours'
REMOTE="$("${SSH[@]}" "ls -1 ${REMOTE_DIR}/run-v5*.json 2>/dev/null | head -1")"
if [[ -z "${REMOTE}" ]]; then
  echo "no run-v5*.json on Mac under ${REMOTE_DIR}" >&2
  exit 1
fi
BASE="$(basename "${REMOTE}")"
"${SCP[@]}" "kooshas-laptop:${REMOTE}" "${OUT}/${BASE}"
echo "wrote ${OUT}/${BASE}"
echo "export BENCH_DATA=${OUT}/${BASE}"

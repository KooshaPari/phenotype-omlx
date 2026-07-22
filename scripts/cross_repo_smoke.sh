#!/usr/bin/env bash
set -euo pipefail

echo "=== Cross-Repo Smoke Test ==="
PASS=0
FAIL=0

# 1. pheno-harness
echo -n "pheno-harness imports... "
if (cd /Users/kooshapari/CodeProjects/Phenotype/pheno-harness && python -c "from bench.types import EnergySource; from bench.executor import Executor; from bench.adapters import ModelAdapter" 2>/dev/null); then
	echo "PASS"
	PASS=$((PASS + 1))
else
	echo "FAIL"
	FAIL=$((FAIL + 1))
fi

# 2. phenotype-omlx (Rust)
echo -n "phenotype-omlx cargo check... "
if (cd /Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx/perf-core && cargo check --quiet 2>/dev/null); then
	echo "PASS"
	PASS=$((PASS + 1))
else
	echo "FAIL"
	FAIL=$((FAIL + 1))
fi

# 3. portage
echo -n "portage imports... "
if (cd /Users/kooshapari/CodeProjects/Phenotype/repos/portage && python -c "import portage" 2>/dev/null); then
	echo "PASS"
	PASS=$((PASS + 1))
else
	echo "FAIL"
	FAIL=$((FAIL + 1))
fi

# 4. Eidolon
echo -n "Eidolon cargo check... "
if (cd /Users/kooshapari/CodeProjects/Phenotype/repos/Eidolon && cargo check --quiet 2>/dev/null); then
	echo "PASS"
	PASS=$((PASS + 1))
else
	echo "FAIL"
	FAIL=$((FAIL + 1))
fi

# 5. Benchora
echo -n "Benchora cargo check... "
if (cd /Users/kooshapari/CodeProjects/Phenotype/repos/Benchora && cargo check --quiet 2>/dev/null); then
	echo "PASS"
	PASS=$((PASS + 1))
else
	echo "FAIL"
	FAIL=$((FAIL + 1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed out of $((PASS + FAIL))"
exit $FAIL

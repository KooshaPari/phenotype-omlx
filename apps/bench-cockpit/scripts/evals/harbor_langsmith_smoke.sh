#!/bin/bash
# Harbor hello-world → LangSmith via harbor-langsmith plugin.
# Uses PORTAGE_ROOT worktree that imports harbor_langsmith cleanly.
# Runtime: Apple Container (`-e apple-container`) — never Docker.
# If Podman is needed elsewhere, Harbor uses `-e docker --ek container_runtime=podman`;
# `podman` itself is not a Harbor environment type.
# shellcheck disable=SC1091
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:/opt/homebrew/Caskroom/miniforge/base/bin:${PATH:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORTAGE="${PORTAGE_ROOT:-/Users/kooshapari/CodeProjects/Phenotype/repos/worktrees/portage/fix-langsmith-importerror}"
HARBOR_ENV="${HARBOR_ENV:-apple-container}"
OUT="${1:-$ROOT/.runs/harbor-smoke}"
mkdir -p "$OUT"

if [[ -f "$ROOT/.env" ]]; then
  eval "$(/usr/bin/grep -E '^(LANGSMITH_API_KEY|LANGSMITH_PROJECT|LANGSMITH_ENDPOINT)=' "$ROOT/.env" | /usr/bin/sed 's/^/export /')"
fi
if [[ -z "${LANGSMITH_API_KEY:-}" ]]; then
  echo "LANGSMITH_API_KEY required" >&2
  exit 1
fi

if [[ "$HARBOR_ENV" != "apple-container" ]]; then
  echo "ERROR: HARBOR_ENV=$HARBOR_ENV is forbidden on this host; use apple-container." >&2
  echo "Podman override elsewhere: -e docker --ek container_runtime=podman" >&2
  exit 1
fi
if ! command -v container >/dev/null 2>&1; then
  echo "ERROR: Apple Container CLI 'container' not on PATH" >&2
  exit 1
fi
if ! container system status >/dev/null 2>&1; then
  echo "Starting Apple Container system…"
  container system start
fi

export PYTHONPATH="$PORTAGE/packages/harbor-langsmith/src${PYTHONPATH:+:$PYTHONPATH}"
export HARBOR_LANGSMITH_DATASET="${HARBOR_LANGSMITH_DATASET:-harbor-hello-world}"
export HARBOR_LANGSMITH_EXPERIMENT="${HARBOR_LANGSMITH_EXPERIMENT:-bench-cockpit-harbor-smoke}"
export HARBOR_LANGSMITH_FAIL_FAST="${HARBOR_LANGSMITH_FAIL_FAST:-true}"

cd "$PORTAGE"
uv run python -c "import harbor_langsmith; print('plugin ok')"

echo "harbor env: $HARBOR_ENV"
uv run harbor run \
  -e "$HARBOR_ENV" \
  -p examples/tasks/hello-world \
  -a oracle \
  -n 1 \
  -y \
  -o "$OUT" \
  --plugin langsmith

echo "done. artifacts: $OUT"
echo "LangSmith dataset: $HARBOR_LANGSMITH_DATASET"

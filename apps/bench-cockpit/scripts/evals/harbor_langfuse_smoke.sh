#!/bin/bash
# Harbor hello-world → Langfuse via harbor-langfuse plugin (primary).
# Runtime: Apple Container (`-e apple-container`) — never Docker.
# shellcheck disable=SC1091
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:/opt/homebrew/Caskroom/miniforge/base/bin:${PATH:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORTAGE="${PORTAGE_ROOT:-}"
HARBOR_ENV="${HARBOR_ENV:-apple-container}"
OUT="${1:-$ROOT/.runs/harbor-langfuse-smoke}"
mkdir -p "$OUT"

if [[ -z "$PORTAGE" ]]; then
  echo "ERROR: PORTAGE_ROOT required (portage checkout with packages/harbor-langfuse)" >&2
  exit 2
fi

if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1090
  set -a && source "$ROOT/.env" && set +a
fi
if [[ -z "${LANGFUSE_PUBLIC_KEY:-}" || -z "${LANGFUSE_SECRET_KEY:-}" ]]; then
  echo "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY required" >&2
  exit 1
fi
export LANGFUSE_BASE_URL="${LANGFUSE_BASE_URL:-https://us.cloud.langfuse.com}"

if [[ "$HARBOR_ENV" != "apple-container" ]]; then
  echo "ERROR: HARBOR_ENV=$HARBOR_ENV is forbidden on this host; use apple-container." >&2
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

if [[ ! -d "$PORTAGE/packages/harbor-langfuse/src" ]]; then
  echo "ERROR: harbor-langfuse missing under $PORTAGE/packages/harbor-langfuse" >&2
  echo "  Pull portage main (feat merged in #478) or set PORTAGE_ROOT to a checkout that has it." >&2
  exit 2
fi

export PYTHONPATH="$PORTAGE/packages/harbor-langfuse/src${PYTHONPATH:+:$PYTHONPATH}"
export HARBOR_LANGFUSE_ENVIRONMENT="${HARBOR_LANGFUSE_ENVIRONMENT:-harbor}"
export HARBOR_LANGFUSE_FAIL_FAST="${HARBOR_LANGFUSE_FAIL_FAST:-true}"

cd "$PORTAGE"
# Entry point `langfuse` requires the package installed (not just PYTHONPATH).
if ! uv run harbor plugins list 2>/dev/null | grep -q langfuse; then
  echo "Installing harbor-langfuse editable so --plugin langfuse resolves…"
  uv pip install -e "$PORTAGE/packages/harbor-langfuse"
fi
uv run python -c "import harbor_langfuse; print('plugin ok', harbor_langfuse.LangfusePlugin)"

echo "harbor env: $HARBOR_ENV base: $LANGFUSE_BASE_URL"
uv run harbor run \
  -e "$HARBOR_ENV" \
  -p examples/tasks/hello-world \
  -a oracle \
  -n 1 \
  -y \
  -o "$OUT" \
  --plugin langfuse

echo "done. artifacts: $OUT"
echo "Langfuse: $LANGFUSE_BASE_URL (sessions tagged lf_runner=harbor)"

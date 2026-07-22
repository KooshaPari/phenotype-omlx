#!/usr/bin/env bash
# scripts/dispatch/sglang.sh
#
# Dispatch entrypoint for the SGLang backend of phenotype-omlx.
#
# Status (2026-07-19): stub. The real dispatch will route requests to
# a side-car SGLang runtime (typically launched via
# `python -m sglang.launch_server --model ...`) and bridge outputs
# back through the kernel-registry. Today no engine is wired, so this
# script prints the command it WOULD run and exits 0.
#
# Probe contract (used by future doctor checks):
#     scripts/dispatch/sglang.sh --help    -> exit 0, prints usage
#     scripts/dispatch/sglang.sh --dry-run -> exit 0, prints dispatch command
#     scripts/dispatch/sglang.sh           -> exit 0, prints dispatch command
#
# Once the SGLang path is real this script will:
#   1. Start (or attach to) the SGLang runtime for MODEL_ID.
#   2. Forward the perf / NIAH workload through
#      `python -m omlx_research.cli inference --backend sglang ...`.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

usage() {
    cat <<USAGE
$SCRIPT_NAME — SGLang dispatch entrypoint (STUB)

Usage:
    $SCRIPT_NAME [--model MODEL_ID] [--lengths LEN ...] [--endpoint URL] [--dry-run]

Options:
    --model MODEL_ID     HuggingFace / local model identifier.
                         Default: from config/smoke_models.json (role=dispatch)
    --lengths LEN ...    Context lengths to run (tokens).
                         Default: 1024 4096 16384
    --endpoint URL       Existing SGLang OpenAI-compatible endpoint to
                         attach to instead of launching a server.
    --dry-run            Print the dispatch command instead of executing it.
    --help               Show this message and exit 0.

Exit codes:
    0   success (or dry-run / --help)
    2   invalid arguments
    64  dispatch target not yet implemented

Examples:
    $SCRIPT_NAME --help
    $SCRIPT_NAME --dry-run
    $SCRIPT_NAME --endpoint http://127.0.0.1:30000/v1
USAGE
}

# Defaults — model from config/smoke_models.json (role=dispatch).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${OMLX_READY_MODEL:-}"
if [[ -z "$MODEL" ]]; then
  MODEL="$(PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" python3 -m omlx_research.smoke_models dispatch)"
fi
LENGTHS="1024 4096 16384"
ENDPOINT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --lengths)
            shift
            LENGTHS=""
            while [[ $# -gt 0 && "$1" != --* ]]; do
                LENGTHS+="$1 "
                shift
            done
            LENGTHS="${LENGTHS% }"
            [[ -z "$LENGTHS" ]] && { echo "$SCRIPT_NAME: --lengths needs at least one value" >&2; exit 2; }
            ;;
        --endpoint)
            ENDPOINT="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "$SCRIPT_NAME: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

DISPATCH_CMD=(
    python3 -m omlx_research.cli inference
        --backend sglang
        --model "$MODEL"
        --lengths $LENGTHS
)

if [[ -n "$ENDPOINT" ]]; then
    DISPATCH_CMD+=(--endpoint "$ENDPOINT")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$SCRIPT_NAME] DRY-RUN — would dispatch via SGLang:"
    printf '    %q ' "${DISPATCH_CMD[@]}"
    echo
    if [[ -z "$ENDPOINT" ]]; then
        echo "[$SCRIPT_NAME] would launch: python3 -m sglang.launch_server --model $MODEL --port 30000"
    else
        echo "[$SCRIPT_NAME] would attach to existing endpoint: $ENDPOINT"
    fi
    exit 0
fi

echo "[$SCRIPT_NAME] SGLang dispatch is currently a no-op stub (engines not yet wired)."
echo "[$SCRIPT_NAME] model=$MODEL lengths=$LENGTHS endpoint=${ENDPOINT:-<launch locally>}"
echo "[$SCRIPT_NAME] would run: ${DISPATCH_CMD[*]}"
echo "[$SCRIPT_NAME] rerun with --dry-run to silence this banner."
exit 0
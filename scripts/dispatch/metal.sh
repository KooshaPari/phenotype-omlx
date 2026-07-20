#!/usr/bin/env bash
# scripts/dispatch/metal.sh
#
# Dispatch entrypoint for the Metal kernel-registry path of phenotype-omlx.
#
# Status (2026-07-19): stub. The real dispatch will route Metal
# (Apple Silicon) inference through the kernel-registry's compiled
# .metallib and the MLX backend. Today no engine is wired, so this
# script prints the command it WOULD run and exits 0.
#
# Probe contract (used by future doctor checks):
#     scripts/dispatch/metal.sh --help   -> exit 0, prints usage
#     scripts/dispatch/metal.sh --dry-run -> exit 0, prints dispatch command
#     scripts/dispatch/metal.sh          -> exit 0, prints dispatch command
#
# Once the Metal path is real this script will:
#   1. Resolve the model identifier (positional arg or --model flag).
#   2. Locate the .metallib under perf-core/kernel-registry/target/.
#   3. Invoke `python -m omlx_research.cli inference --backend metal ...`
#      with the resolved metallib on MLX_METAL_PATH.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

usage() {
    cat <<USAGE
$SCRIPT_NAME — Metal (Apple Silicon) dispatch entrypoint (STUB)

Usage:
    $SCRIPT_NAME [--model MODEL_ID] [--lengths LEN ...] [--dry-run]

Options:
    --model MODEL_ID    HuggingFace / local model identifier.
                        Default: mlx-community/Qwen2.5-0.5B-Instruct-4bit
    --lengths LEN ...   Context lengths to run (tokens).
                        Default: 1024 4096 16384
    --dry-run           Print the dispatch command instead of executing it.
    --help              Show this message and exit 0.

Exit codes:
    0   success (or dry-run / --help)
    2   invalid arguments
    64  dispatch target not yet implemented

Examples:
    $SCRIPT_NAME --help
    $SCRIPT_NAME --dry-run
    $SCRIPT_NAME --model mlx-community/Llama-3.2-1B-Instruct-4bit
USAGE
}

# Defaults — mirror scripts/perf_turboquant.py so the surfaced
# command stays consistent across dispatch entrypoints.
MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LENGTHS="1024 4096 16384"
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
            # Positional model argument.
            MODEL="$1"
            shift
            ;;
    esac
done

# The dispatch command we WOULD run. Kept identical between --dry-run
# and the real path so users can preview exactly what would happen.
DISPATCH_CMD=(
    python3 -m omlx_research.cli inference
        --backend metal
        --model "$MODEL"
        --lengths $LENGTHS
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$SCRIPT_NAME] DRY-RUN — would dispatch Metal inference via kernel-registry:"
    printf '    %q ' "${DISPATCH_CMD[@]}"
    echo
    echo "[$SCRIPT_NAME] target metallib: perf-core/kernel-registry/target/*.metallib (not built yet)"
    exit 0
fi

echo "[$SCRIPT_NAME] Metal dispatch is currently a no-op stub (engines not yet wired)."
echo "[$SCRIPT_NAME] model=$MODEL lengths=$LENGTHS"
echo "[$SCRIPT_NAME] would run: ${DISPATCH_CMD[*]}"
echo "[$SCRIPT_NAME] rerun with --dry-run to silence this banner."
exit 0
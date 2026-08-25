#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 GPU BATCH_STAMP SWANLAB_GROUP VARIANT|INIT|SLUG [...]" >&2
    exit 2
fi

GPU_ID="$1"
BATCH_STAMP="$2"
SWANLAB_GROUP="$3"
shift 3
csig_require_allowed_gpu "$GPU_ID"

RUNNER="$REPO_ROOT/tools/run_structure_candidate_experiment.sh"
for configuration in "$@"; do
    IFS='|' read -r variant init_mode slug extra <<<"$configuration"
    if [[ -z "$variant" || -z "$init_mode" || -z "$slug" || -n "${extra:-}" ]]; then
        echo "Invalid queued configuration: $configuration" >&2
        exit 2
    fi
    if [[ "$init_mode" != "scratch" ]]; then
        echo "Scratch-only policy rejects queue initialization: $init_mode" >&2
        exit 2
    fi

    echo "Starting queued experiment on GPU $GPU_ID: $slug"
    env \
        STRUCTURE_ADAPTER_LR=0.001 \
        STRUCTURE_BASE_LR_MULT=5.0 \
        STRUCTURE_SEED="${STRUCTURE_SEED:-47}" \
        STRUCTURE_INIT_MODE="$init_mode" \
        STRUCTURE_USE_SWANLAB="${STRUCTURE_USE_SWANLAB:-0}" \
        STRUCTURE_SWANLAB_MODE="${STRUCTURE_SWANLAB_MODE:-offline}" \
        THRESHOLD_GRID="${THRESHOLD_GRID:-0.10:0.95:0.01}" \
        bash "$RUNNER" \
            "$GPU_ID" "$variant" "$slug" "$BATCH_STAMP" "$SWANLAB_GROUP"
done

echo "GPU $GPU_ID queue completed."

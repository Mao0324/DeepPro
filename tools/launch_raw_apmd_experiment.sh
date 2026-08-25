#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
RUNNER="$REPO_ROOT/tools/run_structure_candidate_experiment.sh"
GPU_ID="${1:-0}"

if [[ $# -gt 1 || ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [GPU_ID]" >&2
    exit 2
fi
csig_require_allowed_gpu "$GPU_ID"

used_memory="$({
    nvidia-smi --id="$GPU_ID" --query-gpu=memory.used \
        --format=csv,noheader,nounits
} | tr -d '[:space:]')"
if [[ ! "$used_memory" =~ ^[0-9]+$ ]]; then
    echo "Could not read GPU $GPU_ID memory usage." >&2
    exit 1
fi
if [[ "$used_memory" -gt 1024 ]]; then
    echo "GPU $GPU_ID is not idle: ${used_memory} MiB used." >&2
    echo "No experiment was started." >&2
    exit 1
fi

batch_stamp="$(date -u +%Y-%m-%d_%H-%M-%S)"
run_date="${batch_stamp%%_*}"
session="csig_apmd_g${GPU_ID}_${batch_stamp}"
screen_log_root="$REPO_ROOT/log/sem_seg/$run_date/_structure_screen_logs"
screen_log="$screen_log_root/${session}.log"
swanlab_group="f1_raw_apmd_${batch_stamp}"
mkdir -p "$screen_log_root"

if screen -S "$session" -Q select . >/dev/null 2>&1; then
    echo "Screen session already exists: $session" >&2
    exit 1
fi
if [[ -e "$screen_log" ]]; then
    echo "Screen log already exists: $screen_log" >&2
    exit 1
fi

screen -dmS "$session" -L -Logfile "$screen_log" \
    env \
        STRUCTURE_ADAPTER_LR=0.001 \
        STRUCTURE_BASE_LR_MULT=5.0 \
        STRUCTURE_SEED=46 \
        THRESHOLD_GRID=0.10:0.95:0.01 \
    bash "$RUNNER" \
        "$GPU_ID" raw_apmd brtd3_raw_apmd "$batch_stamp" "$swanlab_group"

echo "Started raw APMD experiment on GPU $GPU_ID: $session"
echo "Attach: screen -r $session"
echo "Screen log: $screen_log"
echo "The runner will produce and validate a tracked submission ZIP."

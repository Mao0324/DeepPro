#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 BATCH_STAMP SWANLAB_RUN_ID" >&2
    exit 2
fi

BATCH_STAMP="$1"
SWANLAB_RUN_ID="$2"
GPU_SPEC="0,1,2"
VARIANT="raw_apmd_hybrid_rms_scratch_init"
STRUCTURE_SEED="${STRUCTURE_SEED:-47}"
SLUG="hrms_scratch_init_ddp3_seed${STRUCTURE_SEED}"
RUN_DATE="${BATCH_STAMP%%_*}"
EXPERIMENT="$REPO_ROOT/log/sem_seg/$RUN_DATE/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${SLUG}_E100"
CHECKPOINT="$EXPERIMENT/checkpoints/latest_model.pth"
SCREEN_LOG_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE/_priority_ddp3_screen_logs"
SESSION="csig_priority_ddp3_fast_s${STRUCTURE_SEED}_${BATCH_STAMP}"
SCREEN_LOG="$SCREEN_LOG_ROOT/${SESSION}.log"
SWANLAB_GROUP="priority_scratch_ddp3_seed${STRUCTURE_SEED}_${BATCH_STAMP}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-/home/user/.swanlab/.netrc}"
RUNNER="$REPO_ROOT/tools/run_structure_candidate_experiment.sh"

csig_require_allowed_gpus "$GPU_SPEC"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Resume checkpoint is unavailable: $CHECKPOINT" >&2
    exit 1
fi
if screen -S "$SESSION" -Q select . >/dev/null 2>&1; then
    echo "Screen session already exists: $SESSION" >&2
    exit 1
fi

mapfile -t GPU_MEMORY < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
for gpu in 0 1 2; do
    used_memory="${GPU_MEMORY[$gpu]//[[:space:]]/}"
    if [[ ! "$used_memory" =~ ^[0-9]+$ || "$used_memory" -gt 1024 ]]; then
        echo "GPU $gpu is not idle: ${used_memory:-unknown} MiB used" >&2
        exit 1
    fi
done

mkdir -p "$SCREEN_LOG_ROOT"
screen -dmS "$SESSION" -L -Logfile "$SCREEN_LOG" \
    env \
        STRUCTURE_ADAPTER_LR=0.001 \
        STRUCTURE_BASE_LR_MULT=5.0 \
        STRUCTURE_SEED="$STRUCTURE_SEED" \
        STRUCTURE_BATCH_SIZE=18 \
        STRUCTURE_GRAD_ACCUM_STEPS=1 \
        STRUCTURE_EVAL_INTERVAL=5 \
        STRUCTURE_INIT_MODE=scratch \
        STRUCTURE_RESUME_MODE=auto \
        STRUCTURE_RESUME_CHECKPOINT="$CHECKPOINT" \
        STRUCTURE_USE_SWANLAB=1 \
        STRUCTURE_SWANLAB_MODE=cloud \
        STRUCTURE_SWANLAB_ID="$SWANLAB_RUN_ID" \
        STRUCTURE_SWANLAB_RESUME=allow \
        SWANLAB_CREDENTIAL_FILE="$SWANLAB_CREDENTIAL_FILE" \
        THRESHOLD_GRID="${THRESHOLD_GRID:-0.10:0.95:0.01}" \
    bash "$RUNNER" \
        "$GPU_SPEC" "$VARIANT" "$SLUG" "$BATCH_STAMP" "$SWANLAB_GROUP"

echo "Resumed priority DDP run with distributed validation every 5 epochs."
echo "session=$SESSION"
echo "screen_log=$SCREEN_LOG"

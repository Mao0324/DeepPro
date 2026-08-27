#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
RUNNER="$REPO_ROOT/tools/run_feedbacksts_f1_experiment.sh"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 ORIGINAL_BATCH_STAMP SWANLAB_RUN_ID" >&2
    exit 2
fi

GPU_SPEC="0,1,2"
BATCH_STAMP="$1"
SWANLAB_RUN_ID="$2"
FEEDBACK_SEED="${FEEDBACK_SEED:-47}"
FEEDBACK_SEQ_LEN="${FEEDBACK_SEQ_LEN:-40}"
FEEDBACK_BATCH_SIZE="${FEEDBACK_BATCH_SIZE:-6}"
FEEDBACK_LEARNING_RATE="${FEEDBACK_LEARNING_RATE:-0.0005}"
SLUG="feedbacksts_l${FEEDBACK_SEQ_LEN}_t2_recallaug_ddp3_seed${FEEDBACK_SEED}"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
EXPERIMENT="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__FeedbackSTS-F1-${SLUG}_E100"
CHECKPOINT="$EXPERIMENT/checkpoints/latest_model.pth"
RESUME_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
SESSION="csig_feedbacksts_b${FEEDBACK_BATCH_SIZE}_resume_${RESUME_STAMP}"
SCREEN_LOG_ROOT="$DAY_ROOT/_feedbacksts_screen_logs"
SCREEN_LOG="$SCREEN_LOG_ROOT/${SESSION}.log"
SWANLAB_GROUP="feedbacksts_f1_ddp3_seed${FEEDBACK_SEED}_${BATCH_STAMP}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-/home/user/.swanlab/.netrc}"

csig_require_allowed_gpus "$GPU_SPEC"
if [[ "$FEEDBACK_SEQ_LEN" -ne 40 ]]; then
    echo "F1-priority resume requires FEEDBACK_SEQ_LEN=40." >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Resume checkpoint is unavailable: $CHECKPOINT" >&2
    exit 1
fi
if [[ "$FEEDBACK_BATCH_SIZE" -le 0 || $((FEEDBACK_BATCH_SIZE % 3)) -ne 0 ]]; then
    echo "Global batch must be positive and divisible by three." >&2
    exit 2
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
        FEEDBACK_SEED="$FEEDBACK_SEED" \
        FEEDBACK_SEQ_LEN="$FEEDBACK_SEQ_LEN" \
        FEEDBACK_BATCH_SIZE="$FEEDBACK_BATCH_SIZE" \
        FEEDBACK_LEARNING_RATE="$FEEDBACK_LEARNING_RATE" \
        FEEDBACK_EPOCHS="${FEEDBACK_EPOCHS:-100}" \
        FEEDBACK_EVAL_INTERVAL="${FEEDBACK_EVAL_INTERVAL:-5}" \
        FEEDBACK_RESUME_MODE=auto \
        FEEDBACK_SWANLAB_ID="$SWANLAB_RUN_ID" \
        FEEDBACK_SWANLAB_RESUME=must \
        SWANLAB_CREDENTIAL_FILE="$SWANLAB_CREDENTIAL_FILE" \
        THRESHOLD_GRID="${THRESHOLD_GRID:-0.02:0.80:0.01}" \
    bash "$RUNNER" "$GPU_SPEC" "$SLUG" "$BATCH_STAMP" "$SWANLAB_GROUP"

echo "Resumed FeedbackSTS DDP run: $SESSION"
echo "global_batch=$FEEDBACK_BATCH_SIZE per_gpu_batch=$((FEEDBACK_BATCH_SIZE / 3))"
echo "sequence_length=$FEEDBACK_SEQ_LEN"
echo "learning_rate=$FEEDBACK_LEARNING_RATE"
echo "checkpoint=$CHECKPOINT"
echo "screen_log=$SCREEN_LOG"

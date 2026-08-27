#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
RUNNER="$REPO_ROOT/tools/run_feedbacksts_f1_experiment.sh"

DRY_RUN=0
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    [[ "$1" == "--dry-run" ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
    DRY_RUN=1
fi

GPU_SPEC="0,1,2"
FEEDBACK_SEED="${FEEDBACK_SEED:-47}"
FEEDBACK_SEQ_LEN="${FEEDBACK_SEQ_LEN:-40}"
SLUG="feedbacksts_l${FEEDBACK_SEQ_LEN}_t2_recallaug_ddp3_seed${FEEDBACK_SEED}"
BATCH_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
EXPERIMENT="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__FeedbackSTS-F1-${SLUG}_E100"
SCREEN_LOG_ROOT="$DAY_ROOT/_feedbacksts_screen_logs"
SESSION="csig_feedbacksts_ddp3_s${FEEDBACK_SEED}_${BATCH_STAMP}"
SCREEN_LOG="$SCREEN_LOG_ROOT/${SESSION}.log"
SWANLAB_GROUP="feedbacksts_f1_ddp3_seed${FEEDBACK_SEED}_${BATCH_STAMP}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-/home/user/.swanlab/.netrc}"

csig_require_allowed_gpus "$GPU_SPEC"
if [[ "$FEEDBACK_SEQ_LEN" -ne 40 ]]; then
    echo "F1-priority launcher requires FEEDBACK_SEQ_LEN=40." >&2
    exit 2
fi
if [[ -e "$EXPERIMENT" ]]; then
    echo "Refusing to reuse existing experiment: $EXPERIMENT" >&2
    exit 1
fi
if [[ -z "${SWANLAB_API_KEY:-}" && ! -r "$SWANLAB_CREDENTIAL_FILE" ]]; then
    echo "SwanLab cloud credential is unavailable: $SWANLAB_CREDENTIAL_FILE" >&2
    exit 1
fi

echo "model=DeepPro-FeedbackSTS scratch_only=1"
echo "gpus=$GPU_SPEC world_size=3 global_batch=${FEEDBACK_BATCH_SIZE:-6}"
echo "sequence_length=$FEEDBACK_SEQ_LEN (baseline temporal context)"
echo "learning_rate=${FEEDBACK_LEARNING_RATE:-0.0005}"
echo "experiment=$EXPERIMENT"
echo "expected_zip=$EXPERIMENT/submission/submit_${SLUG}_best_proxy_f1.zip"
echo "swanlab_group=$SWANLAB_GROUP"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run only."
    exit 0
fi

mapfile -t GPU_MEMORY < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
if [[ "${#GPU_MEMORY[@]}" -lt 3 ]]; then
    echo "Expected physical GPUs 0-2, found ${#GPU_MEMORY[@]} device(s)." >&2
    exit 1
fi
for gpu in 0 1 2; do
    used_memory="${GPU_MEMORY[$gpu]//[[:space:]]/}"
    if [[ ! "$used_memory" =~ ^[0-9]+$ || "$used_memory" -gt 1024 ]]; then
        echo "GPU $gpu is not idle: ${used_memory:-unknown} MiB used" >&2
        exit 1
    fi
done

mkdir -p "$SCREEN_LOG_ROOT"
if screen -S "$SESSION" -Q select . >/dev/null 2>&1; then
    echo "Screen session already exists: $SESSION" >&2
    exit 1
fi
screen -dmS "$SESSION" -L -Logfile "$SCREEN_LOG" \
    env \
        FEEDBACK_SEED="$FEEDBACK_SEED" \
        FEEDBACK_SEQ_LEN="$FEEDBACK_SEQ_LEN" \
        FEEDBACK_BATCH_SIZE="${FEEDBACK_BATCH_SIZE:-6}" \
        FEEDBACK_LEARNING_RATE="${FEEDBACK_LEARNING_RATE:-0.0005}" \
        FEEDBACK_EPOCHS="${FEEDBACK_EPOCHS:-100}" \
        FEEDBACK_EVAL_INTERVAL="${FEEDBACK_EVAL_INTERVAL:-5}" \
        SWANLAB_CREDENTIAL_FILE="$SWANLAB_CREDENTIAL_FILE" \
        THRESHOLD_GRID="${THRESHOLD_GRID:-0.02:0.80:0.01}" \
    bash "$RUNNER" "$GPU_SPEC" "$SLUG" "$BATCH_STAMP" "$SWANLAB_GROUP"

echo "Started FeedbackSTS three-GPU DDP run: $SESSION"
echo "screen_log=$SCREEN_LOG"

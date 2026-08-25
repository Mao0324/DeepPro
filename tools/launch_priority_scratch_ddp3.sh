#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
RUNNER="$REPO_ROOT/tools/run_structure_candidate_experiment.sh"

DRY_RUN=0
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    if [[ "$1" != "--dry-run" ]]; then
        echo "Usage: $0 [--dry-run]" >&2
        exit 2
    fi
    DRY_RUN=1
fi

GPU_SPEC="0,1,2"
VARIANT="raw_apmd_hybrid_rms_scratch_init"
STRUCTURE_SEED="${STRUCTURE_SEED:-47}"
SLUG="hrms_scratch_init_ddp3_seed${STRUCTURE_SEED}"
BATCH_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
EXPERIMENT="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${SLUG}_E100"
SCREEN_LOG_ROOT="$DAY_ROOT/_priority_ddp3_screen_logs"
SESSION="csig_priority_ddp3_s${STRUCTURE_SEED}_${BATCH_STAMP}"
SCREEN_LOG="$SCREEN_LOG_ROOT/${SESSION}.log"
SWANLAB_GROUP="priority_scratch_ddp3_seed${STRUCTURE_SEED}_${BATCH_STAMP}"
STRUCTURE_USE_SWANLAB="${STRUCTURE_USE_SWANLAB:-1}"
STRUCTURE_SWANLAB_MODE="${STRUCTURE_SWANLAB_MODE:-cloud}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-/home/user/.swanlab/.netrc}"
STRUCTURE_BATCH_SIZE="${STRUCTURE_BATCH_SIZE:-18}"
STRUCTURE_GRAD_ACCUM_STEPS="${STRUCTURE_GRAD_ACCUM_STEPS:-1}"

csig_require_allowed_gpus "$GPU_SPEC"
if [[ "$STRUCTURE_BATCH_SIZE" -le 0 || "$STRUCTURE_BATCH_SIZE" -ne 18 ]]; then
    echo "Priority DDP run requires global batch 18 (6 samples per GPU)." >&2
    exit 2
fi
if [[ "$STRUCTURE_GRAD_ACCUM_STEPS" -ne 1 ]]; then
    echo "Priority DDP run requires gradient accumulation 1." >&2
    exit 2
fi
if [[ "$STRUCTURE_USE_SWANLAB" != "1" || "$STRUCTURE_SWANLAB_MODE" != "cloud" ]]; then
    echo "Priority DDP run requires SwanLab cloud logging." >&2
    exit 2
fi
if [[ -z "${SWANLAB_API_KEY:-}" && ! -r "$SWANLAB_CREDENTIAL_FILE" ]]; then
    echo "SwanLab cloud credential is unavailable: $SWANLAB_CREDENTIAL_FILE" >&2
    exit 1
fi
if [[ -e "$EXPERIMENT" ]]; then
    echo "Refusing to reuse existing experiment: $EXPERIMENT" >&2
    exit 1
fi

echo "variant=$VARIANT"
echo "gpus=$GPU_SPEC world_size=3 global_batch=18 per_gpu_batch=6"
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

mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_structure_pipeline_status"
if screen -S "$SESSION" -Q select . >/dev/null 2>&1; then
    echo "Screen session already exists: $SESSION" >&2
    exit 1
fi
screen -dmS "$SESSION" -L -Logfile "$SCREEN_LOG" \
    env \
        STRUCTURE_ADAPTER_LR=0.001 \
        STRUCTURE_BASE_LR_MULT=5.0 \
        STRUCTURE_SEED="$STRUCTURE_SEED" \
        STRUCTURE_BATCH_SIZE="$STRUCTURE_BATCH_SIZE" \
        STRUCTURE_GRAD_ACCUM_STEPS="$STRUCTURE_GRAD_ACCUM_STEPS" \
        STRUCTURE_EVAL_INTERVAL="${STRUCTURE_EVAL_INTERVAL:-5}" \
        STRUCTURE_INIT_MODE=scratch \
        STRUCTURE_USE_SWANLAB="$STRUCTURE_USE_SWANLAB" \
        STRUCTURE_SWANLAB_MODE="$STRUCTURE_SWANLAB_MODE" \
        SWANLAB_CREDENTIAL_FILE="$SWANLAB_CREDENTIAL_FILE" \
        THRESHOLD_GRID="${THRESHOLD_GRID:-0.10:0.95:0.01}" \
    bash "$RUNNER" \
        "$GPU_SPEC" "$VARIANT" "$SLUG" "$BATCH_STAMP" "$SWANLAB_GROUP"

echo "Started priority three-GPU DDP run: $SESSION"
echo "screen_log=$SCREEN_LOG"

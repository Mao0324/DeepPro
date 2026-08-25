#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
QUEUE_RUNNER="$REPO_ROOT/tools/run_structure_candidate_queue.sh"

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

BATCH_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
SCREEN_LOG_ROOT="$DAY_ROOT/_structure_screen_logs"
STRUCTURE_SEED="${STRUCTURE_SEED:-47}"
SWANLAB_GROUP="f1_hybrid_rms_scratch_only_seed${STRUCTURE_SEED}_${BATCH_STAMP}"
GPU_IDS=(0 1 2)

# GPU|variant|initialization|slug. All runs are scratch-only. Repeated GPU IDs
# are sequential queue items, never concurrent jobs on the same device.
CONFIGURATIONS=(
    "0|raw_apmd_hybrid_rms|scratch|brtd3_raw_apmd_hybrid_rms_scratch_seed${STRUCTURE_SEED}"
    "1|raw_apmd_hybrid_rms_motion_detrend|scratch|brtd3_raw_apmd_hybrid_rms_motion_detrend_scratch_seed${STRUCTURE_SEED}"
    "2|raw_apmd_hybrid_rms_multiscale_contrast|scratch|brtd3_raw_apmd_hybrid_rms_multiscale_contrast_scratch_seed${STRUCTURE_SEED}"
    "0|raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast|scratch|brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast_scratch_seed${STRUCTURE_SEED}"
)

if [[ "$DRY_RUN" -eq 0 ]]; then
    mapfile -t GPU_MEMORY < <(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    )
    if [[ "${#GPU_MEMORY[@]}" -lt 4 ]]; then
        echo "Expected physical GPUs 0-3, found ${#GPU_MEMORY[@]} device(s)." >&2
        exit 1
    fi
    for gpu in "${GPU_IDS[@]}"; do
        csig_require_allowed_gpu "$gpu"
        used_memory="${GPU_MEMORY[$gpu]//[[:space:]]/}"
        if [[ ! "$used_memory" =~ ^[0-9]+$ || "$used_memory" -gt 1024 ]]; then
            echo "GPU $gpu is not idle: ${used_memory:-unknown} MiB used" >&2
            exit 1
        fi
    done
    mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_structure_pipeline_status"
fi

declare -A QUEUE_COUNTS=([0]=0 [1]=0 [2]=0)
for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu variant init_mode slug <<<"$configuration"
    csig_require_allowed_gpu "$gpu"
    QUEUE_COUNTS[$gpu]=$((QUEUE_COUNTS[$gpu] + 1))
    wave="${QUEUE_COUNTS[$gpu]}"
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${slug}_E100"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "wave=$wave GPU=$gpu init=$init_mode variant=$variant experiment=$experiment"
    elif [[ -e "$experiment" ]]; then
        echo "Refusing to reuse existing experiment: $experiment" >&2
        exit 1
    fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run only: 4 scratch experiments, 3 GPU queues, allowed GPUs $CSIG_ALLOWED_GPU_IDS."
    exit 0
fi

for gpu in "${GPU_IDS[@]}"; do
    session="csig_hrms_queue_g${gpu}_s${STRUCTURE_SEED}_${BATCH_STAMP}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    if screen -S "$session" -Q select . >/dev/null 2>&1; then
        echo "Screen session already exists: $session" >&2
        exit 1
    fi
    if [[ -e "$screen_log" ]]; then
        echo "Screen log already exists: $screen_log" >&2
        exit 1
    fi
done

for gpu in "${GPU_IDS[@]}"; do
    queue=()
    for configuration in "${CONFIGURATIONS[@]}"; do
        IFS='|' read -r assigned_gpu variant init_mode slug <<<"$configuration"
        if [[ "$assigned_gpu" == "$gpu" ]]; then
            queue+=("$variant|$init_mode|$slug")
        fi
    done
    session="csig_hrms_queue_g${gpu}_s${STRUCTURE_SEED}_${BATCH_STAMP}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    screen -dmS "$session" -L -Logfile "$screen_log" \
        env \
            STRUCTURE_SEED="$STRUCTURE_SEED" \
            STRUCTURE_USE_SWANLAB="${STRUCTURE_USE_SWANLAB:-0}" \
            STRUCTURE_SWANLAB_MODE="${STRUCTURE_SWANLAB_MODE:-offline}" \
        bash "$QUEUE_RUNNER" \
            "$gpu" "$BATCH_STAMP" "$SWANLAB_GROUP" "${queue[@]}"
    echo "Started sequential queue on GPU $gpu: $session (${#queue[@]} experiments)"
    echo "  attach: screen -r $session"
    echo "  log:    $screen_log"
done

echo "Started scratch-only queues on GPUs 0,1,2; at most three experiments run concurrently."

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

BATCH_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
SCREEN_LOG_ROOT="$DAY_ROOT/_structure_screen_logs"
SWANLAB_GROUP="f1_raw_apmd_struct3_2seed_${BATCH_STAMP}"

# GPU|seed|variant|short tag|slug
CONFIGURATIONS=(
    "2|47|raw_apmd_channel_rms|chrms|brtd3_raw_apmd_channel_rms_seed47"
    "3|49|raw_apmd_channel_rms|chrms|brtd3_raw_apmd_channel_rms_seed49"
    "4|47|raw_apmd_motion_detrend|mdet|brtd3_raw_apmd_motion_detrend_seed47"
    "5|49|raw_apmd_motion_detrend|mdet|brtd3_raw_apmd_motion_detrend_seed49"
    "6|47|raw_apmd_multiscale_contrast|msc|brtd3_raw_apmd_multiscale_contrast_seed47"
    "7|49|raw_apmd_multiscale_contrast|msc|brtd3_raw_apmd_multiscale_contrast_seed49"
)

if [[ "$DRY_RUN" -eq 0 ]]; then
    mapfile -t GPU_MEMORY < <(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    )
    if [[ "${#GPU_MEMORY[@]}" -lt 8 ]]; then
        echo "Expected at least 8 GPUs, found ${#GPU_MEMORY[@]}" >&2
        exit 1
    fi
    mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_structure_pipeline_status"
fi

for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu seed variant tag slug <<<"$configuration"
    csig_require_allowed_gpu "$gpu"
    session="csig_opt_g${gpu}_${tag}_s${seed}_${BATCH_STAMP}"
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${slug}_E100"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "GPU $gpu seed=$seed variant=$variant experiment=$experiment"
        continue
    fi
    used_memory="${GPU_MEMORY[$gpu]//[[:space:]]/}"
    if [[ ! "$used_memory" =~ ^[0-9]+$ || "$used_memory" -gt 1024 ]]; then
        echo "GPU $gpu is not idle: ${used_memory:-unknown} MiB used" >&2
        exit 1
    fi
    if [[ -e "$experiment" ]]; then
        echo "Refusing to reuse existing experiment: $experiment" >&2
        exit 1
    fi
    if screen -S "$session" -Q select . >/dev/null 2>&1; then
        echo "Screen session already exists: $session" >&2
        exit 1
    fi
    if [[ -e "$screen_log" ]]; then
        echo "Screen log already exists: $screen_log" >&2
        exit 1
    fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run only; no training was started."
    exit 0
fi

for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu seed variant tag slug <<<"$configuration"
    session="csig_opt_g${gpu}_${tag}_s${seed}_${BATCH_STAMP}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    screen -dmS "$session" -L -Logfile "$screen_log" \
        env \
            STRUCTURE_ADAPTER_LR=0.001 \
            STRUCTURE_BASE_LR_MULT=5.0 \
            STRUCTURE_SEED="$seed" \
            THRESHOLD_GRID=0.10:0.95:0.01 \
        bash "$RUNNER" \
            "$gpu" "$variant" "$slug" "$BATCH_STAMP" "$SWANLAB_GROUP"
    echo "Started GPU $gpu seed $seed variant $variant: $session"
    echo "  attach: screen -r $session"
    echo "  log:    $screen_log"
done

echo "Six paired optimization runs were started on GPUs 2-7."
echo "Seeds: 47/49 per variant; Top-5 postprocessing and ZIP validation enabled."

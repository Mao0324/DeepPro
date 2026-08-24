#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
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
STRUCTURE_SEED="${STRUCTURE_SEED:-47}"
SWANLAB_GROUP="f1_hybrid_rms_init_ablation_seed${STRUCTURE_SEED}_${BATCH_STAMP}"

# GPU|variant|short tag|init mode|slug
CONFIGURATIONS=(
    "0|raw_apmd_hybrid_rms|hrms|pretrained|brtd3_raw_apmd_hybrid_rms_pretrained_seed${STRUCTURE_SEED}"
    "1|raw_apmd_hybrid_rms|hrms|scratch|brtd3_raw_apmd_hybrid_rms_scratch_seed${STRUCTURE_SEED}"
    "2|raw_apmd_hybrid_rms_motion_detrend|hmdet|pretrained|brtd3_raw_apmd_hybrid_rms_motion_detrend_pretrained_seed${STRUCTURE_SEED}"
    "3|raw_apmd_hybrid_rms_motion_detrend|hmdet|scratch|brtd3_raw_apmd_hybrid_rms_motion_detrend_scratch_seed${STRUCTURE_SEED}"
    "4|raw_apmd_hybrid_rms_multiscale_contrast|hmsc|pretrained|brtd3_raw_apmd_hybrid_rms_multiscale_contrast_pretrained_seed${STRUCTURE_SEED}"
    "5|raw_apmd_hybrid_rms_multiscale_contrast|hmsc|scratch|brtd3_raw_apmd_hybrid_rms_multiscale_contrast_scratch_seed${STRUCTURE_SEED}"
    "6|raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast|hfull|pretrained|brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast_pretrained_seed${STRUCTURE_SEED}"
    "7|raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast|hfull|scratch|brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast_scratch_seed${STRUCTURE_SEED}"
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
    IFS='|' read -r gpu variant tag init_mode slug <<<"$configuration"
    session="csig_hrms_g${gpu}_${tag}_${init_mode}_s${STRUCTURE_SEED}_${BATCH_STAMP}"
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${slug}_E100"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "GPU $gpu seed=$STRUCTURE_SEED init=$init_mode variant=$variant experiment=$experiment"
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
    IFS='|' read -r gpu variant tag init_mode slug <<<"$configuration"
    session="csig_hrms_g${gpu}_${tag}_${init_mode}_s${STRUCTURE_SEED}_${BATCH_STAMP}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    screen -dmS "$session" -L -Logfile "$screen_log" \
        env \
            STRUCTURE_ADAPTER_LR=0.001 \
            STRUCTURE_BASE_LR_MULT=5.0 \
            STRUCTURE_SEED="$STRUCTURE_SEED" \
            STRUCTURE_INIT_MODE="$init_mode" \
            STRUCTURE_USE_SWANLAB=1 \
            STRUCTURE_SWANLAB_MODE=cloud \
            THRESHOLD_GRID=0.10:0.95:0.01 \
        bash "$RUNNER" \
            "$gpu" "$variant" "$slug" "$BATCH_STAMP" "$SWANLAB_GROUP"
    echo "Started GPU $gpu seed $STRUCTURE_SEED init $init_mode variant $variant: $session"
    echo "  attach: screen -r $session"
    echo "  log:    $screen_log"
done

echo "Eight hybrid-RMS initialization-ablation runs were started on GPUs 0-7."
echo "Seed: $STRUCTURE_SEED; pretrained/scratch paired per structure; Top-5 postprocessing and ZIP validation enabled."

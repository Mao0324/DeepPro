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
SWANLAB_GROUP="f1_structure_round2_${BATCH_STAMP}"

CONFIGURATIONS=(
    "0|second_order|brtd3_second_order"
    "1|tdc_dual_stream|brtd3_tdc_dual_stream"
    "2|lfp_shallow|brtd3_lfp_shallow"
    "3|lfp_deep|brtd3_lfp_deep"
    "4|global_align|brtd3_global_align"
    "5|local_align|brtd3_local_align"
    "6|multiscale_head|brtd3_multiscale_head"
    "7|bidirectional|brtd3_bidirectional"
)

if [[ "$DRY_RUN" -eq 0 ]]; then
    mapfile -t GPU_MEMORY < <(
        nvidia-smi --query-gpu=memory.used \
            --format=csv,noheader,nounits
    )
    if [[ "${#GPU_MEMORY[@]}" -lt 8 ]]; then
        echo "Expected 8 GPUs, found ${#GPU_MEMORY[@]}" >&2
        exit 1
    fi
    for gpu in {0..7}; do
        used="${GPU_MEMORY[$gpu]//[[:space:]]/}"
        if [[ "$used" -gt 1024 ]]; then
            echo "GPU $gpu is not idle: ${used} MiB used" >&2
            echo "Wait for the current experiments before launching round 2." >&2
            exit 1
        fi
    done
    mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_structure_pipeline_status"
fi

for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu variant slug <<<"$configuration"
    session="csig_struct_g${gpu}_${variant}"
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${slug}_E100"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "GPU $gpu  variant=$variant  experiment=$experiment"
        continue
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
    IFS='|' read -r gpu variant slug <<<"$configuration"
    session="csig_struct_g${gpu}_${variant}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    screen -dmS "$session" -L -Logfile "$screen_log" \
        bash "$RUNNER" \
            "$gpu" "$variant" "$slug" "$BATCH_STAMP" "$SWANLAB_GROUP"
    echo "started GPU $gpu: $session"
    echo "  attach: screen -r $session"
    echo "  log:    $screen_log"
done

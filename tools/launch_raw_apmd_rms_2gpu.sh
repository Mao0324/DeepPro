#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
RUNNER="$REPO_ROOT/tools/run_structure_candidate_experiment.sh"
BATCH_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
SCREEN_LOG_ROOT="$DAY_ROOT/_structure_screen_logs"
SWANLAB_GROUP="f1_raw_apmd_rms_2seed_${BATCH_STAMP}"

# Fixed paired seeds: both have DeepPro and Raw-APMD baselines from 8/13-8/17.
# GPU|seed|slug
CONFIGURATIONS=(
    "0|47|brtd3_raw_apmd_rms_seed47"
    "1|49|brtd3_raw_apmd_rms_seed49"
)

mapfile -t GPU_MEMORY < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
if [[ "${#GPU_MEMORY[@]}" -lt 2 ]]; then
    echo "Expected physical GPUs 0-3, found ${#GPU_MEMORY[@]}" >&2
    exit 1
fi

mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_structure_pipeline_status"
for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu seed slug <<<"$configuration"
    csig_require_allowed_gpu "$gpu"
    used_memory="${GPU_MEMORY[$gpu]//[[:space:]]/}"
    session="csig_apmd_rms_g${gpu}_seed${seed}_${BATCH_STAMP}"
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${slug}_E100"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
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

for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu seed slug <<<"$configuration"
    session="csig_apmd_rms_g${gpu}_seed${seed}_${BATCH_STAMP}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    screen -dmS "$session" -L -Logfile "$screen_log" \
        env \
            STRUCTURE_ADAPTER_LR=0.001 \
            STRUCTURE_BASE_LR_MULT=5.0 \
            STRUCTURE_SEED="$seed" \
            THRESHOLD_GRID=0.10:0.95:0.01 \
        bash "$RUNNER" \
            "$gpu" raw_apmd_rms "$slug" "$BATCH_STAMP" "$SWANLAB_GROUP"
    echo "Started GPU $gpu seed $seed: $session"
    echo "  attach: screen -r $session"
    echo "  log:    $screen_log"
done

echo "Both Raw-APMD-RMS runs were started."
echo "Seeds: 47 and 49; adapter LR: 0.001; backbone LR: 0.005."
echo "Each runner uses early stopping and generates a validated tracked ZIP."

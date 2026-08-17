#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
RUNNER="$REPO_ROOT/tools/run_structure_candidate_experiment.sh"
BATCH_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
SCREEN_LOG_ROOT="$DAY_ROOT/_structure_screen_logs"
SWANLAB_GROUP="f1_raw_apmd_3seed_${BATCH_STAMP}"

# GPU|seed|slug
CONFIGURATIONS=(
    "0|46|brtd3_raw_apmd_seed46"
    "1|47|brtd3_raw_apmd_seed47"
    "2|49|brtd3_raw_apmd_seed49"
)

mapfile -t GPU_MEMORY < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
if [[ "${#GPU_MEMORY[@]}" -lt 3 ]]; then
    echo "Expected at least 3 GPUs, found ${#GPU_MEMORY[@]}" >&2
    exit 1
fi

mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_structure_pipeline_status"
for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu seed slug <<<"$configuration"
    used_memory="${GPU_MEMORY[$gpu]//[[:space:]]/}"
    session="csig_apmd_g${gpu}_seed${seed}_${BATCH_STAMP}"
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
    session="csig_apmd_g${gpu}_seed${seed}_${BATCH_STAMP}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    screen -dmS "$session" -L -Logfile "$screen_log" \
        env \
            STRUCTURE_ADAPTER_LR=0.001 \
            STRUCTURE_BASE_LR_MULT=5.0 \
            STRUCTURE_SEED="$seed" \
            THRESHOLD_GRID=0.10:0.95:0.01 \
        bash "$RUNNER" \
            "$gpu" raw_apmd "$slug" "$BATCH_STAMP" "$SWANLAB_GROUP"
    echo "Started GPU $gpu seed $seed: $session"
    echo "  attach: screen -r $session"
    echo "  log:    $screen_log"
done

echo "All three Raw-APMD runs were started."
echo "Early stopping: eval_f1, patience=30, min_delta=1e-4, start_epoch=15."
echo "Each runner will generate and validate a tracked submission ZIP."

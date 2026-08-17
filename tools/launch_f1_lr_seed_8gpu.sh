#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
RUNNER="$REPO_ROOT/tools/run_valid_frame_architecture_experiment.sh"
EXPERIMENT_DAY="2026-08-13"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$EXPERIMENT_DAY"
SCREEN_LOG_ROOT="$DAY_ROOT/_screen_logs"
SWANLAB_GROUP="f1_lr_seed_calibration_8gpu"
THRESHOLD_GRID="0.10:0.95:0.01"
mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_pipeline_status"

# GPU|MODEL|SLUG|BACKGROUND|ADAPTIVE|GATE|ADAPTER_LR|BASE_MULT|SEED
CONFIGURATIONS=(
    "0|DeepPro-Plus|deeppro_seed47|1|1|1|0.005|1.0|47"
    "1|DeepPro-Plus|deeppro_seed48|1|1|1|0.005|1.0|48"
    "2|DeepPro-Plus|deeppro_seed49|1|1|1|0.005|1.0|49"
    "3|DeepPro-Plus_BRTD2|brtdv2_nogate_base0p0005|1|1|0|0.001|0.5|46"
    "4|DeepPro-Plus_BRTD2|brtdv2_nogate_base0p0010|1|1|0|0.001|1.0|46"
    "5|DeepPro-Plus_BRTD2|brtdv2_nogate_base0p0025|1|1|0|0.001|2.5|46"
    "6|DeepPro-Plus_BRTD2|brtdv2_nogate_base0p0050|1|1|0|0.001|5.0|46"
    "7|DeepPro-Plus_BRTD2|brtdv2_nogate_adapter0p0005_base0p0050|1|1|0|0.0005|10.0|46"
)

for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu model slug background adaptive gate lr base_lr_mult seed \
        <<<"$configuration"
    session="csig2b_g${gpu}_${slug}"
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__${EXPERIMENT_DAY}__ValidFrames-F1OHEM-${slug}_E100"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
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
    IFS='|' read -r gpu model slug background adaptive gate lr base_lr_mult seed \
        <<<"$configuration"
    session="csig2b_g${gpu}_${slug}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    screen -dmS "$session" -L -Logfile "$screen_log" \
        env \
            EXPERIMENT_DAY="$EXPERIMENT_DAY" \
            EXPERIMENT_SEED="$seed" \
            THRESHOLD_GRID="$THRESHOLD_GRID" \
            SWANLAB_GROUP="$SWANLAB_GROUP" \
        bash "$RUNNER" \
            "$gpu" "$model" "$slug" "$background" "$adaptive" "$gate" \
            "$lr" "$base_lr_mult"
    echo "started GPU $gpu: $session"
    echo "  seed=$seed adapter_lr=$lr base_lr_multiplier=$base_lr_mult"
    echo "  attach: screen -r $session"
    echo "  log:    $screen_log"
done

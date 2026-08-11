#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
RUNNER="$REPO_ROOT/tools/run_valid_frame_architecture_experiment.sh"
DAY_ROOT="$REPO_ROOT/log/sem_seg/2026-08-11"
SCREEN_LOG_ROOT="$DAY_ROOT/_screen_logs"
mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_pipeline_status"

CONFIGURATIONS=(
    "0|DeepPro-Plus|deeppro_validmask|1|1|1|0.005|1.0"
    "1|DeepPro-Plus_BRTD|brtd_full_validmask|1|1|1|0.001|0.1"
    "2|DeepPro-Plus_BRTD2|brtdv2_full_validmask|1|1|1|0.001|0.1"
    "3|DeepPro-Plus_BRTD2|brtdv2_no_background|0|1|1|0.001|0.1"
    "4|DeepPro-Plus_BRTD2|brtdv2_fixed_router|1|0|1|0.001|0.1"
    "5|DeepPro-Plus_BRTD2|brtdv2_no_gate|1|1|0|0.001|0.1"
    "6|DeepPro-Plus_BRTD|brtd_no_background|0|1|1|0.001|0.1"
    "7|DeepPro-Plus_BRTD|brtd_fixed_router|1|0|1|0.001|0.1"
)

for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu model slug background adaptive gate lr base_lr_mult \
        <<<"$configuration"
    session="csig2a_g${gpu}_${slug}"
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__2026-08-11__ValidFrames-F1OHEM-${slug}_E100"
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
    IFS='|' read -r gpu model slug background adaptive gate lr base_lr_mult \
        <<<"$configuration"
    session="csig2a_g${gpu}_${slug}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    screen -dmS "$session" -L -Logfile "$screen_log" \
        bash "$RUNNER" \
            "$gpu" "$model" "$slug" "$background" "$adaptive" "$gate" \
            "$lr" "$base_lr_mult"
    echo "started GPU $gpu: $session"
    echo "  attach: screen -r $session"
    echo "  log:    $screen_log"
done

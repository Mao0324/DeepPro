#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
SWANLAB_BIN="/home/devbox/project/model/miniconda3/envs/sjyPID/bin/swanlab"
RUNNER="$REPO_ROOT/tools/run_f1_ohem_resume_to100.sh"
STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
SCREEN_LOG_DIR="$REPO_ROOT/screen_logs"

"$SWANLAB_BIN" verify
mkdir -p "$SCREEN_LOG_DIR"

launch() {
    local model="$1"
    local gpu="$2"
    local experiment="$3"
    local learning_rate="$4"
    local base_lr_mult="$5"
    local slug="$6"
    local swanlab_id="$7"
    local screen_name="f1ohem100_"$slug"_"$STAMP
    local screen_log="$SCREEN_LOG_DIR/"$screen_name".log"

    if [[ ! -f "$REPO_ROOT/log/sem_seg/$experiment/checkpoints/latest_model.pth" ]]; then
        echo "Missing resume checkpoint for $experiment" >&2
        exit 1
    fi
    if [[ -e "$screen_log" ]]; then
        echo "Refusing to overwrite screen log: $screen_log" >&2
        exit 1
    fi

    screen -L -Logfile "$screen_log" -dmS "$screen_name" \
        bash "$RUNNER" \
        "$model" "$gpu" "$experiment" "$learning_rate" \
        "$base_lr_mult" "$slug" "$swanlab_id"

    printf '%s\tGPU%s\t%s\t%s\n' \
        "$screen_name" "$gpu" "$experiment" "$screen_log"
}

launch \
    "DeepPro-Plus" 1 \
    "SatVideoIRSDT_v1__2026-08-06_11-24-50__F1-Calibrated-OHEM-Pretrained_DeepPro-Plus_DataL40_E50" \
    0.005 1.0 "deeppro_plus" "w888xy63dtsncfibqei8t"

launch \
    "DeepPro-Plus_BRTD" 3 \
    "SatVideoIRSDT_v1__2026-08-06_11-24-50__F1-Calibrated-OHEM-Pretrained_DeepPro-Plus_BRTD_DataL40_E50" \
    0.001 0.1 "brtd" "7romh6ob2nj90bdtrssr1"

launch \
    "DeepPro-Plus_BRTD2" 4 \
    "SatVideoIRSDT_v1__2026-08-06_11-24-50__F1-Calibrated-OHEM-Pretrained_DeepPro-Plus_BRTD2_DataL40_E50" \
    0.001 0.1 "brtdv2" "h2qfq9sgjf033ie14sh22"

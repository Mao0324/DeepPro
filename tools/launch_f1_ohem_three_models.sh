#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
SWANLAB_BIN="/home/devbox/project/model/miniconda3/envs/sjyPID/bin/swanlab"
RUNNER="$REPO_ROOT/tools/run_f1_ohem_experiment.sh"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [cloud|local|offline]" >&2
    exit 2
fi
SWANLAB_MODE="cloud"
if [[ $# -eq 1 ]]; then
    SWANLAB_MODE="$1"
fi
if [[ -n "${F1_OHEM_STAMP:-}" ]]; then
    STAMP="$F1_OHEM_STAMP"
else
    STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi
SCREEN_LOG_DIR="$REPO_ROOT/screen_logs"

if [[ "$SWANLAB_MODE" != "cloud" && "$SWANLAB_MODE" != "local" && "$SWANLAB_MODE" != "offline" ]]; then
    echo "SwanLab mode must be cloud, local, or offline." >&2
    exit 2
fi
if [[ "$SWANLAB_MODE" == "cloud" ]]; then
    "$SWANLAB_BIN" verify
fi

mkdir -p "$SCREEN_LOG_DIR"

launch() {
    local model="$1"
    local gpu="$2"
    local learning_rate="$3"
    local base_lr_mult="$4"
    local slug="$5"
    local screen_name="f1ohem_"$slug"_"$STAMP
    local experiment="SatVideoIRSDT_v1__"$STAMP"__F1-Calibrated-OHEM-Pretrained_"$model"_DataL40_E50"
    local experiment_dir="$REPO_ROOT/log/sem_seg/$experiment"
    local screen_log="$SCREEN_LOG_DIR/"$screen_name".log"

    if [[ -e "$experiment_dir" || -e "$screen_log" ]]; then
        echo "Refusing to overwrite existing experiment or screen log: $experiment" >&2
        exit 1
    fi

    screen -L -Logfile "$screen_log" -dmS "$screen_name" \
        bash "$RUNNER" \
        "$model" "$gpu" "$experiment" "$learning_rate" \
        "$base_lr_mult" "$slug" "$SWANLAB_MODE"

    printf '%s\tGPU%s\t%s\t%s\n' \
        "$screen_name" "$gpu" "$experiment" "$screen_log"
}

launch "DeepPro-Plus" 1 0.005 1.0 "deeppro_plus"
launch "DeepPro-Plus_BRTD" 3 0.001 0.1 "brtd"
launch "DeepPro-Plus_BRTD2" 4 0.001 0.1 "brtdv2"

#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
STREAMER="$REPO_ROOT/tools/stream_training_log_to_swanlab.py"
RUN_DATE="2026-08-22"
BATCH_STAMP="2026-08-22_08-27-32"
SEED=47
MIN_EPOCH=9
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
STATUS_ROOT="$DAY_ROOT/_structure_pipeline_status"
SIDECAR_ROOT="$DAY_ROOT/_swanlab_sidecars"
GROUP="f1_hybrid_rms_init_ablation_seed47_${BATCH_STAMP}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-}"

if [[ -z "${SWANLAB_API_KEY:-}" && -n "$SWANLAB_CREDENTIAL_FILE" && -r "$SWANLAB_CREDENTIAL_FILE" ]]; then
    SWANLAB_API_KEY="$(awk '$1 == "password" {print $2; exit}' "$SWANLAB_CREDENTIAL_FILE")"
    export SWANLAB_API_KEY
fi
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    echo "SwanLab credential is unavailable." >&2
    exit 1
fi

# GPU|variant|tag|initialization|slug
CONFIGURATIONS=(
    "0|raw_apmd_hybrid_rms|hrms|pretrained|brtd3_raw_apmd_hybrid_rms_pretrained_seed47"
    "1|raw_apmd_hybrid_rms|hrms|scratch|brtd3_raw_apmd_hybrid_rms_scratch_seed47"
    "2|raw_apmd_hybrid_rms_motion_detrend|hmdet|pretrained|brtd3_raw_apmd_hybrid_rms_motion_detrend_pretrained_seed47"
    "3|raw_apmd_hybrid_rms_motion_detrend|hmdet|scratch|brtd3_raw_apmd_hybrid_rms_motion_detrend_scratch_seed47"
    "4|raw_apmd_hybrid_rms_multiscale_contrast|hmsc|pretrained|brtd3_raw_apmd_hybrid_rms_multiscale_contrast_pretrained_seed47"
    "5|raw_apmd_hybrid_rms_multiscale_contrast|hmsc|scratch|brtd3_raw_apmd_hybrid_rms_multiscale_contrast_scratch_seed47"
    "6|raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast|hfull|pretrained|brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast_pretrained_seed47"
    "7|raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast|hfull|scratch|brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast_scratch_seed47"
)

mkdir -p "$SIDECAR_ROOT"
for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu variant tag init_mode slug <<<"$configuration"
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${slug}_E100"
    source_log="$experiment/logs/DeepPro-Plus_BRTD3.txt"
    status_file="$STATUS_ROOT/${slug}.status"
    state_file="$SIDECAR_ROOT/${slug}.json"
    sidecar_log="$SIDECAR_ROOT/${slug}.log"
    session="swan_hrms_g${gpu}_${tag}_${init_mode}_s47"
    run_id="hrms-g${gpu}-${tag}-${init_mode}-s47-0822082732"
    if [[ ! -f "$source_log" || ! -f "$status_file" ]]; then
        echo "Missing source log or status for $slug" >&2
        exit 1
    fi
    if screen -S "$session" -Q select . >/dev/null 2>&1; then
        echo "Sidecar session already exists: $session" >&2
        exit 1
    fi
    screen -dmS "$session" -L -Logfile "$sidecar_log" \
        env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
        "$PYTHON_BIN" -u "$STREAMER" \
            --log-file "$source_log" \
            --status-file "$status_file" \
            --state-file "$state_file" \
            --project CSIG2026-DeepPro \
            --group "$GROUP" \
            --run-name "$slug" \
            --run-id "$run_id" \
            --variant "$variant" \
            --init-mode "$init_mode" \
            --seed "$SEED" \
            --min-epoch "$MIN_EPOCH"
    echo "Started SwanLab sidecar: $session (first epoch $MIN_EPOCH)"
done

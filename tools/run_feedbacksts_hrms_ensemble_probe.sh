#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"

PROBE_GPU="${PROBE_GPU:-0}"
csig_require_allowed_gpu "$PROBE_GPU"

BASELINE_EXPERIMENT_NAME="2026-08-22/SatVideoIRSDT_v1__2026-08-22_08-27-32__F1OHEM-brtd3_raw_apmd_hybrid_rms_scratch_seed47_E100"
FEEDBACK_EXPERIMENT_NAME="2026-08-27/SatVideoIRSDT_v1__2026-08-27_03-23-00__FeedbackSTS-F1-feedbacksts_l40_t2_recallaug_ddp3_seed47_E100"
BASELINE_EPOCH=86
FEEDBACK_EPOCHS=(30 35)

BASELINE_DIR="$REPO_ROOT/log/sem_seg/$BASELINE_EXPERIMENT_NAME"
FEEDBACK_DIR="$REPO_ROOT/log/sem_seg/$FEEDBACK_EXPERIMENT_NAME"
PROBE_ROOT="$FEEDBACK_DIR/postprocess/ensemble_probe_hrms86_feedback30_35"
PREDICTION_ROOT="$PROBE_ROOT/probabilities"
RESULT_ROOT="$PROBE_ROOT/results"
LOG_ROOT="$PROBE_ROOT/logs"
SUBMISSION_ROOT="$PROBE_ROOT/submission"
LOCK_FILE="$FEEDBACK_DIR/postprocess/.ensemble_probe.lock"
STATUS_FILE="$PROBE_ROOT/status.txt"

mkdir -p "$PROBE_ROOT" "$PREDICTION_ROOT" "$RESULT_ROOT" \
    "$LOG_ROOT" "$SUBMISSION_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another FeedbackSTS/Hybrid-RMS ensemble probe is already running." >&2
    exit 1
fi

on_error() {
    local exit_code=$?
    printf 'FAILED exit=%s time=%s\n' \
        "$exit_code" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
    exit "$exit_code"
}
trap on_error ERR

require_scratch_experiment() {
    local model_log="$1"
    local namespace_line
    local initialization_line
    namespace_line="$(grep -m1 'Namespace(' "$model_log" || true)"
    initialization_line="$(
        grep -m1 'from random weights; no base checkpoint loaded' \
            "$model_log" || true
    )"
    if [[ "$namespace_line" != *"base_ckpt=''"* \
          || "$namespace_line" != *"spatial_ckpt=''"* \
          || "$namespace_line" != *"st_ckpt=''"* \
          || -z "$initialization_line" ]]; then
        echo "Scratch-only verification failed for $model_log" >&2
        exit 1
    fi
}

require_empty_prediction_root() {
    local prediction_dir="$1"
    if [[ -d "$prediction_dir" ]] && find "$prediction_dir" -mindepth 1 \
            -print -quit | grep -q .; then
        echo "Refusing to overwrite prediction directory: $prediction_dir" >&2
        exit 1
    fi
}

BASELINE_MODEL_LOG="$BASELINE_DIR/logs/DeepPro-Plus_BRTD3.txt"
FEEDBACK_MODEL_LOG="$FEEDBACK_DIR/logs/DeepPro-FeedbackSTS.txt"
require_scratch_experiment "$BASELINE_MODEL_LOG"
require_scratch_experiment "$FEEDBACK_MODEL_LOG"

if [[ ! -f "$BASELINE_DIR/checkpoints/epoch_${BASELINE_EPOCH}_model.pth" ]]; then
    echo "Missing Hybrid-RMS epoch $BASELINE_EPOCH checkpoint." >&2
    exit 1
fi
for epoch_number in "${FEEDBACK_EPOCHS[@]}"; do
    if [[ ! -f "$FEEDBACK_DIR/checkpoints/epoch_${epoch_number}_model.pth" ]]; then
        echo "Missing FeedbackSTS epoch $epoch_number checkpoint." >&2
        exit 1
    fi
done

printf 'RUNNING gpu=%s scratch_only=1 time=%s\n' \
    "$PROBE_GPU" "$(date --iso-8601=seconds)" >"$STATUS_FILE"

cd "$REPO_ROOT"

BASELINE_PREDICTIONS="$PREDICTION_ROOT/hrms_epoch_${BASELINE_EPOCH}"
require_empty_prediction_root "$BASELINE_PREDICTIONS"
"$PYTHON_BIN" -u test.py \
    --gpu "$PROBE_GPU" \
    --seqlen 40 \
    --datapath "$DATA_ROOT" \
    --dataset SatVideoIRSDT_v1 \
    --logpath "$REPO_ROOT/log" \
    --log_dir "$BASELINE_EXPERIMENT_NAME" \
    --epoch "$BASELINE_EPOCH" \
    --visual \
    --visual_count 0 \
    --visual_dir "$BASELINE_PREDICTIONS" \
    --output_only \
    --test_workers 2 \
    --prefetch_factor 1 \
    --eval_chunk_rows "$TEST_EVAL_CHUNK_ROWS" \
    --amp \
    >"$LOG_ROOT/export_hrms_epoch_${BASELINE_EPOCH}.log" 2>&1

"$PYTHON_BIN" -u tools/centroid_f1_sweep.py \
    --prediction-root "$BASELINE_PREDICTIONS" \
    --data-root "$DATA_ROOT" \
    --split val \
    --thresholds 0.02:0.80:0.01 \
    --min-areas 1,2,3 \
    --match-distance 2 \
    --workers 4 \
    --output-csv "$RESULT_ROOT/hrms_epoch_${BASELINE_EPOCH}.csv" \
    --output-json "$RESULT_ROOT/hrms_epoch_${BASELINE_EPOCH}.json" \
    >"$LOG_ROOT/sweep_hrms_epoch_${BASELINE_EPOCH}.log" 2>&1

ensemble_jsons=()
for epoch_number in "${FEEDBACK_EPOCHS[@]}"; do
    feedback_predictions="$PREDICTION_ROOT/feedback_epoch_${epoch_number}"
    require_empty_prediction_root "$feedback_predictions"
    "$PYTHON_BIN" -u test.py \
        --gpu "$PROBE_GPU" \
        --seqlen 40 \
        --datapath "$DATA_ROOT" \
        --dataset SatVideoIRSDT_v1 \
        --logpath "$REPO_ROOT/log" \
        --log_dir "$FEEDBACK_EXPERIMENT_NAME" \
        --epoch "$epoch_number" \
        --visual \
        --visual_count 0 \
        --visual_dir "$feedback_predictions" \
        --output_only \
        --test_workers 2 \
        --prefetch_factor 1 \
        --amp \
        >"$LOG_ROOT/export_feedback_epoch_${epoch_number}.log" 2>&1

    "$PYTHON_BIN" -u tools/centroid_f1_sweep.py \
        --prediction-root "$feedback_predictions" \
        --data-root "$DATA_ROOT" \
        --split val \
        --thresholds 0.02:0.80:0.01 \
        --min-areas 1,2,3 \
        --match-distance 2 \
        --workers 4 \
        --output-csv "$RESULT_ROOT/feedback_epoch_${epoch_number}.csv" \
        --output-json "$RESULT_ROOT/feedback_epoch_${epoch_number}.json" \
        >"$LOG_ROOT/sweep_feedback_epoch_${epoch_number}.log" 2>&1

    ensemble_json="$RESULT_ROOT/ensemble_hrms${BASELINE_EPOCH}_feedback${epoch_number}.json"
    "$PYTHON_BIN" -u tools/probability_ensemble_sweep.py \
        --prediction-root-a "$BASELINE_PREDICTIONS" \
        --prediction-root-b "$feedback_predictions" \
        --data-root "$DATA_ROOT" \
        --split val \
        --weights-a 0.00:1.00:0.05 \
        --thresholds 0.02:0.80:0.01 \
        --min-areas 1,2,3 \
        --match-distance 2 \
        --workers 4 \
        --output-csv "${ensemble_json%.json}.csv" \
        --output-json "$ensemble_json" \
        >"$LOG_ROOT/sweep_ensemble_feedback_epoch_${epoch_number}.log" 2>&1
    ensemble_jsons+=("$ensemble_json")
done

CENTROID_DIR="$SUBMISSION_ROOT/centroid_best_proxy_f1"
MANIFEST="$RESULT_ROOT/selected_ensemble.json"
"$PYTHON_BIN" -u tools/build_ensemble_submission.py \
    --sweep-json "${ensemble_jsons[@]}" \
    --output-dir "$CENTROID_DIR" \
    --manifest "$MANIFEST" \
    >"$LOG_ROOT/build_ensemble_submission.log" 2>&1

TRACKED_DIR="$SUBMISSION_ROOT/tracked_best_proxy_f1"
ZIP_PATH="$SUBMISSION_ROOT/submit_hrms86_feedbacksts_ensemble_scratch.zip"
"$PYTHON_BIN" -u tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py \
    --input-root "$CENTROID_DIR" \
    --output-dir "$TRACKED_DIR" \
    --max-distance 20 \
    --distance-growth 5 \
    --max-missed 2 \
    --velocity-smoothing 0.5 \
    --zip-output "$ZIP_PATH" \
    >"$LOG_ROOT/track_submission.log" 2>&1

"$PYTHON_BIN" -u tools/validate_submission_zip.py \
    "$ZIP_PATH" --data-root "$DATA_ROOT" --split val \
    | tee "$RESULT_ROOT/submission_validation.txt"
sha256sum "$ZIP_PATH" >"$ZIP_PATH.sha256"
printf 'COMPLETE zip=%s manifest=%s time=%s\n' \
    "$ZIP_PATH" "$MANIFEST" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
echo "Scratch ensemble probe complete: $PROBE_ROOT"
echo "Submission ZIP: $ZIP_PATH"

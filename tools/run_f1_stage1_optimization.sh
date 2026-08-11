#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
PYTHON_BIN="/home/devbox/project/model/miniconda3/envs/sjyPID/bin/python"
DATA_ROOT="/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1"
ANALYSIS_ROOT="$REPO_ROOT/analysis/f1_optimization_2026-08-07"
PROBABILITY_ROOT="$ANALYSIS_ROOT/probabilities"
RESULT_ROOT="$ANALYSIS_ROOT/results"
LOG_ROOT="$ANALYSIS_ROOT/logs"
SUBMISSION_ROOT="$ANALYSIS_ROOT/submission"

DEEPPRO_EXPERIMENT="2026-08-06/SatVideoIRSDT_v1__2026-08-06_11-24-50__F1-Calibrated-OHEM-Pretrained_DeepPro-Plus_DataL40_E50"
BRTD_EXPERIMENT="2026-08-06/SatVideoIRSDT_v1__2026-08-06_11-24-50__F1-Calibrated-OHEM-Pretrained_DeepPro-Plus_BRTD_DataL40_E50"

mkdir -p "$PROBABILITY_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$SUBMISSION_ROOT"
cd "$REPO_ROOT"

run_export() {
    local gpu_id="$1"
    local experiment="$2"
    local epoch="$3"
    local output_dir="$4"
    local epoch_arguments=()
    if [[ "$epoch" != "best" ]]; then
        epoch_arguments=(--epoch "$epoch")
    fi
    "$PYTHON_BIN" -u test.py \
        --gpu "$gpu_id" \
        --seqlen 40 \
        --datapath "$DATA_ROOT" \
        --dataset SatVideoIRSDT_v1 \
        --logpath "$REPO_ROOT/log" \
        --log_dir "$experiment" \
        "${epoch_arguments[@]}" \
        --visual \
        --visual_count 0 \
        --visual_dir "$output_dir" \
        --output_only \
        --test_workers 2 \
        --prefetch_factor 1 \
        --eval_chunk_rows 64
}

run_sweep() {
    local prediction_root="$1"
    local slug="$2"
    "$PYTHON_BIN" -u tools/centroid_f1_sweep.py \
        --prediction-root "$prediction_root" \
        --data-root "$DATA_ROOT" \
        --split val \
        --thresholds 0.25:0.65:0.01 \
        --min-areas 1,2 \
        --match-distance 2 \
        --workers 4 \
        --output-csv "$RESULT_ROOT/${slug}.csv" \
        --output-json "$RESULT_ROOT/${slug}.json"
}

echo "[1/5] Exporting three probability-map sets on GPUs 0, 1 and 2."
run_export 0 "$DEEPPRO_EXPERIMENT" 45 \
    "$PROBABILITY_ROOT/deeppro_epoch45" \
    >"$LOG_ROOT/export_deeppro_epoch45.log" 2>&1 &
pid_deeppro45=$!
run_export 1 "$DEEPPRO_EXPERIMENT" best \
    "$PROBABILITY_ROOT/deeppro_best_e100" \
    >"$LOG_ROOT/export_deeppro_best_e100.log" 2>&1 &
pid_deeppro100=$!
run_export 2 "$BRTD_EXPERIMENT" best \
    "$PROBABILITY_ROOT/brtd_best_e100" \
    >"$LOG_ROOT/export_brtd_best_e100.log" 2>&1 &
pid_brtd=$!
wait "$pid_deeppro45"
wait "$pid_deeppro100"
wait "$pid_brtd"

echo "[2/5] Sweeping threshold and minimum component area for each checkpoint."
run_sweep "$PROBABILITY_ROOT/deeppro_epoch45" deeppro_epoch45 \
    >"$LOG_ROOT/sweep_deeppro_epoch45.log" 2>&1 &
pid_sweep_deeppro45=$!
run_sweep "$PROBABILITY_ROOT/deeppro_best_e100" deeppro_best_e100 \
    >"$LOG_ROOT/sweep_deeppro_best_e100.log" 2>&1 &
pid_sweep_deeppro100=$!
run_sweep "$PROBABILITY_ROOT/brtd_best_e100" brtd_best_e100 \
    >"$LOG_ROOT/sweep_brtd_best_e100.log" 2>&1 &
pid_sweep_brtd=$!
wait "$pid_sweep_deeppro45"
wait "$pid_sweep_deeppro100"
wait "$pid_sweep_brtd"

echo "[3/5] Running coarse DeepPro/BRTD probability-ensemble sweeps."
"$PYTHON_BIN" -u tools/probability_ensemble_sweep.py \
    --prediction-root-a "$PROBABILITY_ROOT/deeppro_epoch45" \
    --prediction-root-b "$PROBABILITY_ROOT/brtd_best_e100" \
    --data-root "$DATA_ROOT" \
    --weights-a 0.5,0.65,0.8,1.0 \
    --thresholds 0.35:0.55:0.025 \
    --min-areas 1,2 \
    --match-distance 2 \
    --workers 8 \
    --output-csv "$RESULT_ROOT/ensemble_deeppro45_brtd.csv" \
    --output-json "$RESULT_ROOT/ensemble_deeppro45_brtd.json" \
    >"$LOG_ROOT/ensemble_deeppro45_brtd.log" 2>&1 &
pid_ensemble45=$!
"$PYTHON_BIN" -u tools/probability_ensemble_sweep.py \
    --prediction-root-a "$PROBABILITY_ROOT/deeppro_best_e100" \
    --prediction-root-b "$PROBABILITY_ROOT/brtd_best_e100" \
    --data-root "$DATA_ROOT" \
    --weights-a 0.5,0.65,0.8,1.0 \
    --thresholds 0.35:0.55:0.025 \
    --min-areas 1,2 \
    --match-distance 2 \
    --workers 8 \
    --output-csv "$RESULT_ROOT/ensemble_deeppro100_brtd.csv" \
    --output-json "$RESULT_ROOT/ensemble_deeppro100_brtd.json" \
    >"$LOG_ROOT/ensemble_deeppro100_brtd.log" 2>&1 &
pid_ensemble100=$!
wait "$pid_ensemble45"
wait "$pid_ensemble100"

echo "[4/5] Building tracked TXT submission from the best proxy-F1 ensemble."
CENTROID_DIR="$SUBMISSION_ROOT/centroid"
TRACKED_DIR="$SUBMISSION_ROOT/tracked"
ZIP_PATH="$SUBMISSION_ROOT/submit_stage1_best_probability_ensemble.zip"
"$PYTHON_BIN" -u tools/build_ensemble_submission.py \
    --sweep-json \
        "$RESULT_ROOT/ensemble_deeppro45_brtd.json" \
        "$RESULT_ROOT/ensemble_deeppro100_brtd.json" \
    --output-dir "$CENTROID_DIR" \
    --manifest "$RESULT_ROOT/selected_ensemble.json" \
    >"$LOG_ROOT/build_ensemble_submission.log" 2>&1
"$PYTHON_BIN" -u tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py \
    --input-root "$CENTROID_DIR" \
    --output-dir "$TRACKED_DIR" \
    --max-distance 20 \
    --distance-growth 5 \
    --max-missed 2 \
    --velocity-smoothing 0.5 \
    --zip-output "$ZIP_PATH" \
    >"$LOG_ROOT/track_ensemble_submission.log" 2>&1

echo "[5/5] Validating archive structure, sequence set, frame counts and fields."
"$PYTHON_BIN" -u tools/validate_submission_zip.py \
    "$ZIP_PATH" --data-root "$DATA_ROOT" --split val \
    | tee "$RESULT_ROOT/submission_validation.txt"
sha256sum "$ZIP_PATH" >"$ZIP_PATH.sha256"
touch "$ANALYSIS_ROOT/COMPLETE"
echo "Stage 1 complete. Submission ZIP: $ZIP_PATH"

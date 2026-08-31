#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"

GPU_ID="${FINAL_TEST_GPU:-0}"
RUN_STAMP="${FINAL_TEST_STAMP:-$(date '+%Y-%m-%d_%H-%M-%S')}"
EXPERIMENT_NAME='2026-08-28/SatVideoIRSDT_v1__2026-08-28_00-38-44__PointCenter-F1-pointcenter_consistency_hrms_scratch_ddp3_seed47_E100'
EXPERIMENT_DIR="$REPO_ROOT/log/sem_seg/$EXPERIMENT_NAME"
MODEL_LOG="$EXPERIMENT_DIR/logs/DeepPro-Plus_BRTD3_PointCenter.txt"
CHECKPOINT="$EXPERIMENT_DIR/checkpoints/epoch_90_model.pth"
EXPECTED_CHECKPOINT_SHA='d724475e089b7d56b6942f017ccdf7f0038c8570d8f6d192c4d1a861c0e4a536'
MODEL_SNAPSHOT="$EXPERIMENT_DIR/DeepPro-Plus_BRTD3_PointCenter.py"
OUTPUT_ROOT="$EXPERIMENT_DIR/final_test/$RUN_STAMP"
PROBABILITY_ROOT="$OUTPUT_ROOT/probabilities"
TRACKED_ROOT="$OUTPUT_ROOT/tracked"
ZIP_PATH="$OUTPUT_ROOT/submit_pointcenter_scratch_epoch90_test_tracked.zip"
STATUS_FILE="$OUTPUT_ROOT/status.txt"
LOCK_FILE="$EXPERIMENT_DIR/final_test/.pointcenter_epoch90.lock"

csig_require_allowed_gpu "$GPU_ID"
if [[ ! -d "$DATA_ROOT/test/img" ]]; then
    echo "Published test image directory is unavailable: $DATA_ROOT/test/img" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Selected checkpoint is unavailable: $CHECKPOINT" >&2
    exit 1
fi
actual_checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
if [[ "$actual_checkpoint_sha" != "$EXPECTED_CHECKPOINT_SHA" ]]; then
    echo "Checkpoint SHA256 mismatch: $actual_checkpoint_sha" >&2
    exit 1
fi
for expected in "base_ckpt=''" "spatial_ckpt=''" "st_ckpt=''"; do
    if ! grep -Fq "$expected" "$MODEL_LOG"; then
        echo "Scratch-only audit failed: missing $expected" >&2
        exit 1
    fi
done
if ! grep -Fq 'from random weights; no base checkpoint loaded.' "$MODEL_LOG"; then
    echo 'Scratch-only audit failed: random initialization was not logged.' >&2
    exit 1
fi

mkdir -p "$EXPERIMENT_DIR/final_test"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo 'Another PointCenter final-test job owns the lock.' >&2
    exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to reuse final-test output: $OUTPUT_ROOT" >&2
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"
printf 'RUNNING gpu=%s checkpoint=epoch90 scratch=1 started=%s\n' \
    "$GPU_ID" "$(date --iso-8601=seconds)" >"$STATUS_FILE"

on_error() {
    local exit_code=$?
    printf 'FAILED exit=%s time=%s\n' \
        "$exit_code" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
    exit "$exit_code"
}
trap on_error ERR

"$PYTHON_BIN" -u test.py \
    --gpu "$GPU_ID" \
    --seqlen 40 \
    --datapath "$DATA_ROOT" \
    --dataset SatVideoIRSDT_v1 \
    --split test \
    --logpath "$REPO_ROOT/log" \
    --log_dir "$EXPERIMENT_NAME" \
    --epoch 90 \
    --visual \
    --visual_count 0 \
    --visual_dir "$PROBABILITY_ROOT" \
    --output_only \
    --test_workers 2 \
    --prefetch_factor 1 \
    --eval_chunk_rows "$TEST_EVAL_CHUNK_ROWS" \
    --amp \
    >"$OUTPUT_ROOT/inference.log" 2>&1

expected_frames=$(find "$DATA_ROOT/test/img" -type f | wc -l)
prediction_frames=$(find "$PROBABILITY_ROOT" -type f -name '*.png' | wc -l)
if [[ "$prediction_frames" -ne "$expected_frames" ]]; then
    echo "Prediction frame mismatch: $prediction_frames versus $expected_frames" >&2
    exit 1
fi

"$PYTHON_BIN" -u tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py \
    --input-root "$PROBABILITY_ROOT" \
    --output-dir "$TRACKED_ROOT" \
    --threshold 0.25 \
    --min-area 2 \
    --max-distance 20 \
    --distance-growth 5 \
    --max-missed 2 \
    --area-weight 0 \
    --max-area-ratio 0 \
    --velocity-smoothing 0.5 \
    --zip-output "$ZIP_PATH" \
    >"$OUTPUT_ROOT/tracking.log" 2>&1

"$PYTHON_BIN" -u tools/validate_submission_zip.py \
    "$ZIP_PATH" --data-root "$DATA_ROOT" --split test \
    | tee "$OUTPUT_ROOT/submission_validation.txt"
sha256sum "$CHECKPOINT" >"$OUTPUT_ROOT/checkpoint.sha256"
sha256sum "$MODEL_SNAPSHOT" >"$OUTPUT_ROOT/model_snapshot.sha256"
sha256sum "$ZIP_PATH" >"$ZIP_PATH.sha256"
printf '%s\n' \
    'model=DeepPro-Plus_BRTD3_PointCenter' \
    'checkpoint=epoch_90_model.pth' \
    'initialization=scratch_only' \
    'validation_proxy_f1=0.7844677137870855' \
    'threshold=0.25' \
    'min_area=2' \
    'tracking=max_distance20,distance_growth5,max_missed2,velocity_smoothing0.5' \
    >"$OUTPUT_ROOT/selection.txt"
touch "$OUTPUT_ROOT/COMPLETE"
printf 'COMPLETE zip=%s time=%s\n' \
    "$ZIP_PATH" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
echo "Final test submission: $ZIP_PATH"

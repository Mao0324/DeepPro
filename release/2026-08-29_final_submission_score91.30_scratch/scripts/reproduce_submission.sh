#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$RELEASE_ROOT/../.." && pwd)"
source "$REPO_ROOT/tools/project_runtime_env.sh"

GPU_ID="${FINAL_SUBMISSION_GPU:-0}"
RUN_STAMP="${FINAL_SUBMISSION_STAMP:-$(date '+%Y-%m-%d_%H-%M-%S')}"
EXPERIMENT_NAME="reproduction/final_submission_score91p30_${RUN_STAMP}"
EXPERIMENT_DIR="$REPO_ROOT/log/sem_seg/$EXPERIMENT_NAME"
OUTPUT_ROOT="$EXPERIMENT_DIR/final_test"
PROBABILITY_ROOT="$OUTPUT_ROOT/probabilities"
TRACKED_ROOT="$OUTPUT_ROOT/tracked_adaptive"
ZIP_PATH="$OUTPUT_ROOT/submit_hrms_scratch_epoch86_adaptive_reproduced.zip"
CHECKPOINT_SOURCE="$RELEASE_ROOT/checkpoint/epoch_86_model.pth"

csig_require_allowed_gpu "$GPU_ID"
[[ -d "$DATA_ROOT/test/img" ]] || {
    echo "Missing published test images: $DATA_ROOT/test/img" >&2
    exit 1
}
[[ ! -e "$EXPERIMENT_DIR" ]] || {
    echo "Refusing to reuse reproduction directory: $EXPERIMENT_DIR" >&2
    exit 1
}

bash "$SCRIPT_DIR/verify_release.sh"
mkdir -p "$EXPERIMENT_DIR/checkpoints" "$OUTPUT_ROOT"
cp "$CHECKPOINT_SOURCE" "$EXPERIMENT_DIR/checkpoints/epoch_86_model.pth"
cp "$RELEASE_ROOT/source_snapshot/DeepPro-Plus_BRTD3.py" "$EXPERIMENT_DIR/"
cp "$RELEASE_ROOT/source_snapshot/structure_adapters.py" "$EXPERIMENT_DIR/"

COMMON_ARGS=(
    --gpu "$GPU_ID"
    --seqlen 40
    --datapath "$DATA_ROOT"
    --dataset SatVideoIRSDT_v1
    --split test
    --logpath "$REPO_ROOT/log"
    --log_dir "$EXPERIMENT_NAME"
    --epoch 86
    --visual
    --visual_count 0
    --visual_dir "$PROBABILITY_ROOT"
    --output_only
    --test_workers 2
    --prefetch_factor 1
    --amp
)

"$PYTHON_BIN" -u "$REPO_ROOT/test.py" \
    "${COMMON_ARGS[@]}" --sequence_stop 218 --eval_chunk_rows 32 \
    >"$OUTPUT_ROOT/inference_part1.log" 2>&1
"$PYTHON_BIN" -u "$REPO_ROOT/test.py" \
    "${COMMON_ARGS[@]}" --sequence_start 218 --eval_chunk_rows 16 \
    --overwrite_outputs >"$OUTPUT_ROOT/inference_part2.log" 2>&1

expected_frames=$(find "$DATA_ROOT/test/img" -type f | wc -l)
prediction_frames=$(find "$PROBABILITY_ROOT" -type f -name '*.png' | wc -l)
[[ "$prediction_frames" -eq "$expected_frames" ]] || {
    echo "Prediction frame mismatch: $prediction_frames versus $expected_frames" >&2
    exit 1
}

"$PYTHON_BIN" -u "$REPO_ROOT/tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py" \
    --input-root "$PROBABILITY_ROOT" --output-dir "$TRACKED_ROOT" \
    --threshold 0.16 --min-area 2 --integer-scale dtype \
    --max-distance 20 --distance-growth 5 --max-missed 2 \
    --min-track-observations 3 --area-weight 0 --max-area-ratio 0 \
    --velocity-smoothing 0.5 >"$OUTPUT_ROOT/tracking.log" 2>&1

for sequence_name in 000204 000205; do
    "$PYTHON_BIN" -u "$REPO_ROOT/tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py" \
        --input-root "$PROBABILITY_ROOT/$sequence_name" \
        --sequence-name "$sequence_name" --output-dir "$TRACKED_ROOT" \
        --threshold 0.96 --min-area 2 --integer-scale dtype \
        --max-distance 20 --distance-growth 5 --max-missed 2 \
        --min-track-observations 4 --area-weight 0 --max-area-ratio 0 \
        --velocity-smoothing 0.5 --overwrite >>"$OUTPUT_ROOT/tracking.log" 2>&1
done

zip -j -q -9 "$ZIP_PATH" "$TRACKED_ROOT"/*.txt
"$PYTHON_BIN" "$REPO_ROOT/tools/validate_submission_zip.py" \
    "$ZIP_PATH" --data-root "$DATA_ROOT" --split test \
    | tee "$OUTPUT_ROOT/submission_validation.txt"
sha256sum "$ZIP_PATH" | tee "$ZIP_PATH.sha256"

printf '%s\n' \
    'initialization=scratch_only' \
    'pretrained_weights=none' \
    'checkpoint=epoch_86_model.pth' \
    'standard_threshold=0.16' \
    'high_resolution_threshold=0.96' \
    >"$OUTPUT_ROOT/provenance.txt"
touch "$OUTPUT_ROOT/COMPLETE"
echo "Reproduced submission: $ZIP_PATH"

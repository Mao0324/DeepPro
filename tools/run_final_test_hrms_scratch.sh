#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"

GPU_ID="${FINAL_TEST_GPU:-0}"
RUN_STAMP="${FINAL_TEST_STAMP:-$(date '+%Y-%m-%d_%H-%M-%S')}"
LARGE_EVAL_CHUNK_ROWS="${FINAL_TEST_LARGE_CHUNK_ROWS:-16}"
EXPERIMENT_NAME='2026-08-22/SatVideoIRSDT_v1__2026-08-22_08-27-32__F1OHEM-brtd3_raw_apmd_hybrid_rms_scratch_seed47_E100'
EXPERIMENT_DIR="$REPO_ROOT/log/sem_seg/$EXPERIMENT_NAME"
CHECKPOINT="$EXPERIMENT_DIR/checkpoints/epoch_86_model.pth"
MODEL_SNAPSHOT="$EXPERIMENT_DIR/DeepPro-Plus_BRTD3.py"
ADAPTER_SNAPSHOT="$EXPERIMENT_DIR/structure_adapters.py"
MODEL_LOG="$EXPERIMENT_DIR/logs/DeepPro-Plus_BRTD3.txt"
EXPECTED_CHECKPOINT_SHA='63d620dedfeab5a58610b90f7a912176368d3e9e48f402364982a052f70373f4'
EXPECTED_MODEL_SHA='bc70225fb63b89d52bf07cb40ee06183bfa1a3824a216a7e51dc819e84c6edd4'
EXPECTED_ADAPTER_SHA='eafa00677946bbb08cb53a716164a58ea6391814650cd54d94bfd4245caf7e25'
OUTPUT_ROOT="$EXPERIMENT_DIR/final_test/$RUN_STAMP"
PROBABILITY_ROOT="$OUTPUT_ROOT/probabilities"
TRACKED_ROOT="$OUTPUT_ROOT/tracked_adaptive_thr0p16_highres0p96"
ZIP_PATH="$OUTPUT_ROOT/submit_hrms_scratch_epoch86_adaptive_thr0p16_highres0p96.zip"
STATUS_FILE="$OUTPUT_ROOT/status.txt"
LOCK_FILE="$EXPERIMENT_DIR/final_test/.hrms_scratch_epoch86_test.lock"

csig_require_allowed_gpu "$GPU_ID"
[[ "$LARGE_EVAL_CHUNK_ROWS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FINAL_TEST_LARGE_CHUNK_ROWS must be a positive integer: $LARGE_EVAL_CHUNK_ROWS" >&2
    exit 1
}
[[ -d "$DATA_ROOT/test/img" ]] || {
    echo "Published test image directory is unavailable: $DATA_ROOT/test/img" >&2
    exit 1
}
for file in "$CHECKPOINT" "$MODEL_SNAPSHOT" "$ADAPTER_SNAPSHOT" "$MODEL_LOG"; do
    [[ -f "$file" ]] || { echo "Required scratch artifact is missing: $file" >&2; exit 1; }
done
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA" ]] || {
    echo 'Scratch checkpoint SHA256 mismatch.' >&2; exit 1;
}
[[ "$(sha256sum "$MODEL_SNAPSHOT" | awk '{print $1}')" == "$EXPECTED_MODEL_SHA" ]] || {
    echo 'Scratch model snapshot SHA256 mismatch.' >&2; exit 1;
}
[[ "$(sha256sum "$ADAPTER_SNAPSHOT" | awk '{print $1}')" == "$EXPECTED_ADAPTER_SHA" ]] || {
    echo 'Scratch adapter snapshot SHA256 mismatch.' >&2; exit 1;
}
for expected in "base_ckpt=''" "spatial_ckpt=''" "st_ckpt=''" "resume='never'"; do
    grep -Fq "$expected" "$MODEL_LOG" || {
        echo "Scratch-only audit failed: missing $expected" >&2; exit 1;
    }
done
grep -Fq 'from random weights; no base checkpoint loaded.' "$MODEL_LOG" || {
    echo 'Scratch-only audit failed: random initialization was not logged.' >&2; exit 1;
}
grep -Fq 'Starting a new experiment from scratch.' "$MODEL_LOG" || {
    echo 'Scratch-only audit failed: scratch training start was not logged.' >&2; exit 1;
}

mkdir -p "$EXPERIMENT_DIR/final_test"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo 'Another scratch epoch86 final-test job owns the lock.' >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "Refusing to reuse final-test output: $OUTPUT_ROOT" >&2; exit 1;
}
mkdir -p "$OUTPUT_ROOT"
printf 'RUNNING stage=inference gpu=%s checkpoint=epoch86 scratch_only=1 started=%s\n' \
    "$GPU_ID" "$(date --iso-8601=seconds)" >"$STATUS_FILE"

on_error() {
    local exit_code=$?
    printf 'FAILED exit=%s time=%s\n' "$exit_code" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
    exit "$exit_code"
}
trap on_error ERR

"$PYTHON_BIN" -u test.py \
    --gpu "$GPU_ID" \
    --seqlen 40 \
    --datapath "$DATA_ROOT" \
    --dataset SatVideoIRSDT_v1 \
    --split test \
    --sequence_stop 218 \
    --logpath "$REPO_ROOT/log" \
    --log_dir "$EXPERIMENT_NAME" \
    --epoch 86 \
    --visual \
    --visual_count 0 \
    --visual_dir "$PROBABILITY_ROOT" \
    --output_only \
    --test_workers 2 \
    --prefetch_factor 1 \
    --eval_chunk_rows "$TEST_EVAL_CHUNK_ROWS" \
    --amp \
    >"$OUTPUT_ROOT/inference_part1.log" 2>&1

# Run the two 1280x1024 sequences in a fresh CUDA process so allocator and
# cuDNN caches from the smaller sequences cannot consume their memory margin.
"$PYTHON_BIN" -u test.py \
    --gpu "$GPU_ID" \
    --seqlen 40 \
    --datapath "$DATA_ROOT" \
    --dataset SatVideoIRSDT_v1 \
    --split test \
    --sequence_start 218 \
    --logpath "$REPO_ROOT/log" \
    --log_dir "$EXPERIMENT_NAME" \
    --epoch 86 \
    --visual \
    --visual_count 0 \
    --visual_dir "$PROBABILITY_ROOT" \
    --output_only \
    --overwrite_outputs \
    --test_workers 2 \
    --prefetch_factor 1 \
    --eval_chunk_rows "$LARGE_EVAL_CHUNK_ROWS" \
    --amp \
    >"$OUTPUT_ROOT/inference_part2.log" 2>&1

expected_sequences=$(find "$DATA_ROOT/test/img" -mindepth 1 -maxdepth 1 -type d | wc -l)
prediction_sequences=$(find "$PROBABILITY_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
expected_frames=$(find "$DATA_ROOT/test/img" -type f | wc -l)
prediction_frames=$(find "$PROBABILITY_ROOT" -type f -name '*.png' | wc -l)
[[ "$prediction_sequences" -eq "$expected_sequences" ]] || {
    echo "Prediction sequence mismatch: $prediction_sequences versus $expected_sequences" >&2
    exit 1
}
[[ "$prediction_frames" -eq "$expected_frames" ]] || {
    echo "Prediction frame mismatch: $prediction_frames versus $expected_frames" >&2
    exit 1
}

printf 'RUNNING stage=tracking threshold=0.16 min_track=3 time=%s\n' \
    "$(date --iso-8601=seconds)" >"$STATUS_FILE"
"$PYTHON_BIN" -u tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py \
    --input-root "$PROBABILITY_ROOT" \
    --output-dir "$TRACKED_ROOT" \
    --threshold 0.16 \
    --min-area 2 \
    --integer-scale dtype \
    --max-distance 20 \
    --distance-growth 5 \
    --max-missed 2 \
    --min-track-observations 3 \
    --area-weight 0 \
    --max-area-ratio 0 \
    --velocity-smoothing 0.5 \
    >"$OUTPUT_ROOT/tracking.log" 2>&1

# The two published 1280x1024 sequences match the high-resolution validation
# regime, whose independently swept optimum is threshold 0.96/min-track 4.
for sequence_name in 000204 000205; do
    [[ -d "$PROBABILITY_ROOT/$sequence_name" ]] || {
        echo "Expected high-resolution sequence is missing: $sequence_name" >&2
        exit 1
    }
    "$PYTHON_BIN" -u tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py \
        --input-root "$PROBABILITY_ROOT/$sequence_name" \
        --sequence-name "$sequence_name" \
        --output-dir "$TRACKED_ROOT" \
        --threshold 0.96 \
        --min-area 2 \
        --integer-scale dtype \
        --max-distance 20 \
        --distance-growth 5 \
        --max-missed 2 \
        --min-track-observations 4 \
        --area-weight 0 \
        --max-area-ratio 0 \
        --velocity-smoothing 0.5 \
        --overwrite \
        >>"$OUTPUT_ROOT/tracking.log" 2>&1
done
zip -j -q -9 "$ZIP_PATH" "$TRACKED_ROOT"/*.txt

"$PYTHON_BIN" -u tools/validate_submission_zip.py \
    "$ZIP_PATH" --data-root "$DATA_ROOT" --split test \
    | tee "$OUTPUT_ROOT/submission_validation.txt"
sha256sum "$CHECKPOINT" >"$OUTPUT_ROOT/checkpoint.sha256"
sha256sum "$MODEL_SNAPSHOT" >"$OUTPUT_ROOT/model_snapshot.sha256"
sha256sum "$ADAPTER_SNAPSHOT" >"$OUTPUT_ROOT/adapter_snapshot.sha256"
sha256sum "$MODEL_LOG" >"$OUTPUT_ROOT/training_log.sha256"
sha256sum "$ZIP_PATH" >"$ZIP_PATH.sha256"
grep -E "Namespace\(|from random weights; no base checkpoint loaded\.|Starting a new experiment from scratch\." \
    "$MODEL_LOG" | head -n 4 >"$OUTPUT_ROOT/scratch_training_evidence.txt"
printf '%s\n' \
    'model=DeepPro-Plus_BRTD3' \
    'checkpoint=epoch_86_model.pth' \
    'initialization=scratch_only' \
    'pretrained_weights=none' \
    'structure=raw_apmd_hybrid_rms' \
    'historical_website_score=86.71' \
    'validation_threshold=0.16' \
    'min_area=2' \
    'min_track_observations=3' \
    'high_resolution_sequences=000204,000205' \
    'high_resolution_threshold=0.96' \
    'high_resolution_min_track_observations=4' \
    'adaptive_tracked_validation_f1=0.780460890' \
    'inference_memory_optimization=release_unused_auxiliary_tensors' \
    'equivalence_check_sequence_000001=byte_identical' \
    'tracking=max_distance20,distance_growth5,max_missed2,velocity_smoothing0.5' \
    >"$OUTPUT_ROOT/provenance.txt"
touch "$OUTPUT_ROOT/COMPLETE"
printf 'COMPLETE zip=%s time=%s\n' "$ZIP_PATH" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
echo "Scratch-only final test submission: $ZIP_PATH"

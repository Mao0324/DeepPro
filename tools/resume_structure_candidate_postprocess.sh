#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
MODEL="DeepPro-Plus_BRTD3"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.10:0.95:0.01}"
EXPECTED_PROBABILITY_FILES="${EXPECTED_PROBABILITY_FILES:-23087}"

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 GPU VARIANT SLUG BATCH_STAMP" >&2
    exit 2
fi

GPU_ID="$1"
csig_require_allowed_gpu "$GPU_ID"
VARIANT="$2"
SLUG="$3"
BATCH_STAMP="$4"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
EXPERIMENT_BASENAME="SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${SLUG}_E100"
EXPERIMENT_NAME="$RUN_DATE/$EXPERIMENT_BASENAME"
EXPERIMENT_DIR="$DAY_ROOT/$EXPERIMENT_BASENAME"
STATUS_DIR="$DAY_ROOT/_structure_pipeline_status"
STATUS_FILE="$STATUS_DIR/${SLUG}.status"
LOCK_FILE="$STATUS_DIR/${SLUG}.postprocess.lock"
POST_ROOT="$EXPERIMENT_DIR/postprocess"
PROBABILITY_ROOT="$POST_ROOT/probabilities"
RESULT_ROOT="$POST_ROOT/results"
LOG_ROOT="$POST_ROOT/logs"
SUBMISSION_ROOT="$EXPERIMENT_DIR/submission"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
MODEL_LOG="$EXPERIMENT_DIR/logs/${MODEL}.txt"
test_amp_arguments=()
if [[ "$TEST_USE_AMP" == "1" ]]; then
    test_amp_arguments=(--amp)
fi

if [[ ! -d "$EXPERIMENT_DIR" || ! -d "$CHECKPOINT_DIR" || ! -f "$MODEL_LOG" ]]; then
    echo "Completed training artifacts are missing: $EXPERIMENT_DIR" >&2
    exit 1
fi

mkdir -p "$STATUS_DIR" "$PROBABILITY_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$SUBMISSION_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Postprocess recovery is already running for $SLUG" >&2
    exit 75
fi
printf 'RUNNING gpu=%s variant=%s resumed_postprocess=%s\n' \
    "$GPU_ID" "$VARIANT" "$(date --iso-8601=seconds)" >"$STATUS_FILE"

on_error() {
    local exit_code=$?
    printf 'FAILED exit=%s stage=postprocess time=%s\n' \
        "$exit_code" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
    exit "$exit_code"
}
trap on_error ERR

sweep_is_complete() {
    local sweep_json="$1"
    local sweep_csv="$2"
    [[ -s "$sweep_csv" ]] || return 1
    "$PYTHON_BIN" - "$sweep_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload["best"]
    float(best["f1"])
    float(best["threshold"])
    int(best["min_area"])
    Path(payload["prediction_root"])
except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

probability_export_is_complete() {
    local probability_dir="$1"
    [[ -d "$probability_dir" ]] || return 1
    local file_count
    file_count="$(find "$probability_dir" -type f -name '*.png' -printf '.' | wc -c)"
    [[ "$file_count" -eq "$EXPECTED_PROBABILITY_FILES" ]]
}

cd "$REPO_ROOT"
mapfile -t CHECKPOINT_SELECTORS < <(
    "$PYTHON_BIN" tools/select_eval_checkpoints.py \
        --log-file "$MODEL_LOG" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --top-k 5 \
        --output-json "$RESULT_ROOT/checkpoint_candidates.json"
)

for selector in "${CHECKPOINT_SELECTORS[@]}"; do
    checkpoint_arguments=()
    if [[ "$selector" == epoch:* ]]; then
        epoch_number="${selector#epoch:}"
        checkpoint_arguments=(--epoch "$epoch_number")
        selector_slug="epoch_${epoch_number}"
    elif [[ "$selector" == best ]]; then
        selector_slug="best"
    else
        echo "Unsupported checkpoint selector: $selector" >&2
        exit 1
    fi

    probability_dir="$PROBABILITY_ROOT/$selector_slug"
    sweep_json="$RESULT_ROOT/${selector_slug}_centroid_f1.json"
    sweep_csv="$RESULT_ROOT/${selector_slug}_centroid_f1.csv"

    if sweep_is_complete "$sweep_json" "$sweep_csv"; then
        echo "[$selector_slug] reusing completed centroid sweep"
    else
        if probability_export_is_complete "$probability_dir"; then
            echo "[$selector_slug] reusing complete probability export"
        else
            if [[ -e "$probability_dir" ]]; then
                case "$probability_dir" in
                    "$PROBABILITY_ROOT"/*) rm -rf -- "$probability_dir" ;;
                    *) echo "Refusing to remove unsafe path: $probability_dir" >&2; exit 1 ;;
                esac
            fi
            echo "[$selector_slug] exporting probabilities"
            "$PYTHON_BIN" -u test.py \
                --gpu "$GPU_ID" \
                --seqlen 40 \
                --datapath "$DATA_ROOT" \
                --dataset SatVideoIRSDT_v1 \
                --logpath "$REPO_ROOT/log" \
                --log_dir "$EXPERIMENT_NAME" \
                "${checkpoint_arguments[@]}" \
                --visual \
                --visual_count 0 \
                --visual_dir "$probability_dir" \
                --output_only \
                --test_workers 2 \
                --prefetch_factor 1 \
                --eval_chunk_rows "$TEST_EVAL_CHUNK_ROWS" \
                "${test_amp_arguments[@]}" \
                >"$LOG_ROOT/export_${selector_slug}.resume.log" 2>&1
            if ! probability_export_is_complete "$probability_dir"; then
                echo "Probability export count mismatch for $selector_slug" >&2
                exit 1
            fi
        fi

        echo "[$selector_slug] running centroid sweep"
        "$PYTHON_BIN" -u tools/centroid_f1_sweep.py \
            --prediction-root "$probability_dir" \
            --data-root "$DATA_ROOT" \
            --split val \
            --thresholds "$THRESHOLD_GRID" \
            --min-areas 1,2,3 \
            --match-distance 2 \
            --workers 4 \
            --output-csv "$sweep_csv" \
            --output-json "$sweep_json" \
            >"$LOG_ROOT/sweep_${selector_slug}.resume.log" 2>&1
    fi

    "$PYTHON_BIN" -u tools/build_single_submission.py \
        --sweep-json "$sweep_json" \
        --output-dir "$SUBMISSION_ROOT/centroid_best_proxy_f1" \
        --manifest "$RESULT_ROOT/selected_submission.json" \
        >"$LOG_ROOT/retain_${selector_slug}.resume.log" 2>&1

    if [[ -d "$probability_dir" ]]; then
        case "$probability_dir" in
            "$PROBABILITY_ROOT"/*) rm -rf -- "$probability_dir" ;;
            *) echo "Refusing to remove unsafe path: $probability_dir" >&2; exit 1 ;;
        esac
    fi
done

TRACKED_DIR="$SUBMISSION_ROOT/tracked_best_proxy_f1"
ZIP_PATH="$SUBMISSION_ROOT/submit_${SLUG}_best_proxy_f1.zip"
"$PYTHON_BIN" -u tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py \
    --input-root "$SUBMISSION_ROOT/centroid_best_proxy_f1" \
    --output-dir "$TRACKED_DIR" \
    --max-distance 20 \
    --distance-growth 5 \
    --max-missed 2 \
    --velocity-smoothing 0.5 \
    --zip-output "$ZIP_PATH" \
    >"$LOG_ROOT/track_submission.resume.log" 2>&1

"$PYTHON_BIN" -u tools/validate_submission_zip.py \
    "$ZIP_PATH" --data-root "$DATA_ROOT" --split val \
    | tee "$RESULT_ROOT/submission_validation.txt"
sha256sum "$ZIP_PATH" >"$ZIP_PATH.sha256"
touch "$EXPERIMENT_DIR/COMPLETE"
printf 'COMPLETE zip=%s time=%s\n' \
    "$ZIP_PATH" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
echo "Postprocess recovery complete: $EXPERIMENT_DIR"
echo "Submission ZIP: $ZIP_PATH"

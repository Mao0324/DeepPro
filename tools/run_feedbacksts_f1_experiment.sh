#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 GPU_OR_COMMA_LIST SLUG BATCH_STAMP SWANLAB_GROUP" >&2
    exit 2
fi

GPU_SPEC="$1"
SLUG="$2"
BATCH_STAMP="$3"
SWANLAB_GROUP="$4"
MODEL="DeepPro-FeedbackSTS"
GPU_NUM=0
IFS=',' read -r -a TRAIN_GPU_IDS <<<"$GPU_SPEC"
GPU_NUM="${#TRAIN_GPU_IDS[@]}"
POSTPROCESS_GPU="${TRAIN_GPU_IDS[0]//[[:space:]]/}"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
EXPERIMENT_BASENAME="SatVideoIRSDT_v1__${BATCH_STAMP}__FeedbackSTS-F1-${SLUG}_E100"
EXPERIMENT_NAME="$RUN_DATE/$EXPERIMENT_BASENAME"
EXPERIMENT_DIR="$DAY_ROOT/$EXPERIMENT_BASENAME"
STATUS_DIR="$DAY_ROOT/_feedbacksts_pipeline_status"
STATUS_FILE="$STATUS_DIR/${SLUG}.status"

FEEDBACK_SEED="${FEEDBACK_SEED:-47}"
FEEDBACK_BATCH_SIZE="${FEEDBACK_BATCH_SIZE:-6}"
FEEDBACK_EPOCHS="${FEEDBACK_EPOCHS:-100}"
FEEDBACK_EVAL_INTERVAL="${FEEDBACK_EVAL_INTERVAL:-5}"
FEEDBACK_LEARNING_RATE="${FEEDBACK_LEARNING_RATE:-0.0005}"
FEEDBACK_RESUME_MODE="${FEEDBACK_RESUME_MODE:-never}"
FEEDBACK_SWANLAB_ID="${FEEDBACK_SWANLAB_ID:-}"
FEEDBACK_SWANLAB_RESUME="${FEEDBACK_SWANLAB_RESUME:-never}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.02:0.80:0.01}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-/home/user/.swanlab/.netrc}"

csig_require_allowed_gpus "$GPU_SPEC"
if [[ "$GPU_SPEC" != "0,1,2" || "$GPU_NUM" -ne 3 ]]; then
    echo "FeedbackSTS priority run requires physical GPUs 0,1,2." >&2
    exit 2
fi
if [[ "$FEEDBACK_BATCH_SIZE" -le 0 || $((FEEDBACK_BATCH_SIZE % GPU_NUM)) -ne 0 ]]; then
    echo "Global batch must be positive and divisible by three." >&2
    exit 2
fi
if [[ -z "${SWANLAB_API_KEY:-}" && -r "$SWANLAB_CREDENTIAL_FILE" ]]; then
    SWANLAB_API_KEY="$(
        awk '$1 == "password" {print $2; exit}' "$SWANLAB_CREDENTIAL_FILE"
    )"
    export SWANLAB_API_KEY
fi
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    echo "SwanLab cloud credential is unavailable." >&2
    exit 1
fi
if [[ "$FEEDBACK_RESUME_MODE" == "auto" && -z "$FEEDBACK_SWANLAB_ID" ]]; then
    echo "A resumed experiment must provide FEEDBACK_SWANLAB_ID." >&2
    exit 2
fi

swanlab_id_arguments=()
if [[ -n "$FEEDBACK_SWANLAB_ID" ]]; then
    swanlab_id_arguments=(--swanlab_id "$FEEDBACK_SWANLAB_ID")
fi
if [[ -e "$EXPERIMENT_DIR" && "$FEEDBACK_RESUME_MODE" != "auto" ]]; then
    echo "Refusing to reuse experiment directory: $EXPERIMENT_DIR" >&2
    exit 1
fi
if [[ "$FEEDBACK_RESUME_MODE" == "auto" && ! -f "$EXPERIMENT_DIR/checkpoints/latest_model.pth" ]]; then
    echo "Resume requested but latest checkpoint is unavailable: $EXPERIMENT_DIR" >&2
    exit 1
fi

mkdir -p "$STATUS_DIR"
printf 'RUNNING gpu=%s model=%s scratch=1 started=%s\n' \
    "$GPU_SPEC" "$MODEL" "$(date --iso-8601=seconds)" >"$STATUS_FILE"

on_error() {
    local exit_code=$?
    printf 'FAILED exit=%s time=%s\n' \
        "$exit_code" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
    exit "$exit_code"
}
trap on_error ERR

cd "$REPO_ROOT"
"$PYTHON_BIN" -u train.py \
    --gpu "$GPU_SPEC" \
    --gpu_num "$GPU_NUM" \
    --model "$MODEL" \
    --dataset SatVideoIRSDT_v1 \
    --datapath "$DATA_ROOT" \
    --savepath "$REPO_ROOT/log" \
    --log_dir "$EXPERIMENT_NAME" \
    --optimizer Adam \
    --learning_rate "$FEEDBACK_LEARNING_RATE" \
    --decay_rate 0.0001 \
    --batch_size "$FEEDBACK_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --epoch "$FEEDBACK_EPOCHS" \
    --early_stopping_patience 6 \
    --early_stopping_min_delta 0.0001 \
    --early_stopping_start_epoch 20 \
    --early_stopping_metric eval_f1 \
    --seqlen 13 \
    --patch_size 128 \
    --sample_rate 0.04 \
    --sequence_augmentation 1 \
    --step_size 10 \
    --lr_decay 0.7 \
    --threshold_eval 0.10 \
    --train_workers 4 \
    --val_workers 2 \
    --prefetch_factor 2 \
    --loss f1_calibrated_ohem \
    --mask_padded_frames 1 \
    --tversky_fp_weight 0.35 \
    --tversky_fn_weight 0.65 \
    --hard_negative_topk 2048 \
    --f1_ohem_dice_weight 0.20 \
    --f1_ohem_hard_weight 0.05 \
    --f1_ohem_negative_ratio 1.5 \
    --f1_ohem_min_negatives 128 \
    --feedback_interval 2 \
    --feedback_alignment_levels 2 \
    --feedback_eval_tile_size 1024 \
    --feedback_eval_tile_overlap 64 \
    --eval_amp 1 \
    --eval_interval "$FEEDBACK_EVAL_INTERVAL" \
    --resume "$FEEDBACK_RESUME_MODE" \
    --seed "$FEEDBACK_SEED" \
    --deterministic 0 \
    --run_test_after_train 0 \
    --use_swanlab 1 \
    --swanlab_project CSIG2026-DeepPro \
    --swanlab_group "$SWANLAB_GROUP" \
    --swanlab_mode cloud \
    --swanlab_resume "$FEEDBACK_SWANLAB_RESUME" \
    "${swanlab_id_arguments[@]}"

POST_ROOT="$EXPERIMENT_DIR/postprocess"
PROBABILITY_ROOT="$POST_ROOT/probabilities"
RESULT_ROOT="$POST_ROOT/results"
LOG_ROOT="$POST_ROOT/logs"
SUBMISSION_ROOT="$EXPERIMENT_DIR/submission"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
MODEL_LOG="$EXPERIMENT_DIR/logs/${MODEL}.txt"
mkdir -p "$PROBABILITY_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$SUBMISSION_ROOT"

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

    "$PYTHON_BIN" -u test.py \
        --gpu "$POSTPROCESS_GPU" \
        --seqlen 13 \
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
        --amp \
        >"$LOG_ROOT/export_${selector_slug}.log" 2>&1

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
        >"$LOG_ROOT/sweep_${selector_slug}.log" 2>&1

    "$PYTHON_BIN" -u tools/build_single_submission.py \
        --sweep-json "$sweep_json" \
        --output-dir "$SUBMISSION_ROOT/centroid_best_proxy_f1" \
        --manifest "$RESULT_ROOT/selected_submission.json" \
        >"$LOG_ROOT/retain_${selector_slug}.log" 2>&1

    rm -rf -- "$probability_dir"
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
    >"$LOG_ROOT/track_submission.log" 2>&1

"$PYTHON_BIN" -u tools/validate_submission_zip.py \
    "$ZIP_PATH" --data-root "$DATA_ROOT" --split val \
    | tee "$RESULT_ROOT/submission_validation.txt"
sha256sum "$ZIP_PATH" >"$ZIP_PATH.sha256"
touch "$EXPERIMENT_DIR/COMPLETE"
printf 'COMPLETE zip=%s time=%s\n' \
    "$ZIP_PATH" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
echo "Experiment complete: $EXPERIMENT_DIR"
echo "Submission ZIP: $ZIP_PATH"

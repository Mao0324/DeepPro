#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
PYTHON_BIN="/home/devbox/project/model/miniconda3/envs/sjyPID/bin/python"
DATA_ROOT="/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1"
BASE_CHECKPOINT="$REPO_ROOT/pretrained/SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth"
MODEL="DeepPro-Plus_BRTD3"

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 GPU VARIANT SLUG BATCH_STAMP SWANLAB_GROUP" >&2
    exit 2
fi

GPU_ID="$1"
VARIANT="$2"
SLUG="$3"
BATCH_STAMP="$4"
SWANLAB_GROUP="$5"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
EXPERIMENT_BASENAME="SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${SLUG}_E100"
EXPERIMENT_NAME="$RUN_DATE/$EXPERIMENT_BASENAME"
EXPERIMENT_DIR="$DAY_ROOT/$EXPERIMENT_BASENAME"
STATUS_DIR="$DAY_ROOT/_structure_pipeline_status"
STATUS_FILE="$STATUS_DIR/${SLUG}.status"

mkdir -p "$STATUS_DIR"
printf 'RUNNING gpu=%s variant=%s started=%s\n' \
    "$GPU_ID" "$VARIANT" "$(date --iso-8601=seconds)" >"$STATUS_FILE"

on_error() {
    local exit_code=$?
    printf 'FAILED exit=%s time=%s\n' \
        "$exit_code" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
    exit "$exit_code"
}
trap on_error ERR

cd "$REPO_ROOT"
"$PYTHON_BIN" -u train.py \
    --gpu "$GPU_ID" \
    --gpu_num 1 \
    --model "$MODEL" \
    --dataset SatVideoIRSDT_v1 \
    --datapath "$DATA_ROOT" \
    --savepath "$REPO_ROOT/log" \
    --log_dir "$EXPERIMENT_NAME" \
    --optimizer Adam \
    --learning_rate 0.001 \
    --base_lr_mult 0.1 \
    --decay_rate 0.0001 \
    --batch_size 20 \
    --epoch 100 \
    --early_stopping_patience 30 \
    --early_stopping_min_delta 0.0001 \
    --early_stopping_start_epoch 15 \
    --early_stopping_metric eval_f1 \
    --seqlen 40 \
    --patch_size 128 \
    --sample_rate 0.04 \
    --step_size 10 \
    --lr_decay 0.7 \
    --threshold_eval 0.5 \
    --train_workers 4 \
    --val_workers 2 \
    --prefetch_factor 2 \
    --loss f1_calibrated_ohem \
    --mask_padded_frames 1 \
    --tversky_fp_weight 0.6 \
    --tversky_fn_weight 0.4 \
    --hard_negative_topk 4096 \
    --base_ckpt "$BASE_CHECKPOINT" \
    --structure_variant "$VARIANT" \
    --structure_bottleneck_channels 8 \
    --structure_max_shift 4.0 \
    --eval_chunk_rows 64 \
    --resume never \
    --seed 46 \
    --deterministic 0 \
    --run_test_after_train 0 \
    --use_swanlab 1 \
    --swanlab_project CSIG2026-DeepPro \
    --swanlab_group "$SWANLAB_GROUP" \
    --swanlab_mode cloud

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
        --top-k 3 \
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
        --eval_chunk_rows 64 \
        >"$LOG_ROOT/export_${selector_slug}.log" 2>&1

    "$PYTHON_BIN" -u tools/centroid_f1_sweep.py \
        --prediction-root "$probability_dir" \
        --data-root "$DATA_ROOT" \
        --split val \
        --thresholds 0.15:0.70:0.01 \
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

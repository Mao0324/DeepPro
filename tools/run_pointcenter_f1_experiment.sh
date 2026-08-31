#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"

GPU_SPEC="0,1,2"
GPU_NUM=3
POSTPROCESS_GPU=0
MODEL="DeepPro-Plus_BRTD3_PointCenter"
POINT_SEED="${POINT_SEED:-47}"
POINT_BATCH_SIZE="${POINT_BATCH_SIZE:-21}"
POINT_EPOCHS="${POINT_EPOCHS:-100}"
POINT_EVAL_INTERVAL="${POINT_EVAL_INTERVAL:-2}"
POINT_RESUME_MODE="${POINT_RESUME_MODE:-never}"
POINT_SWANLAB_ID="${POINT_SWANLAB_ID:-}"
POINT_SWANLAB_RESUME="${POINT_SWANLAB_RESUME:-never}"
POINT_STAMP="${POINT_STAMP:-$(date '+%Y-%m-%d_%H-%M-%S')}"
POINT_SLUG="${POINT_SLUG:-pointcenter_consistency_hrms_scratch_ddp3_seed${POINT_SEED}}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.02:0.80:0.01}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-/home/user/.swanlab/.netrc}"

RUN_DATE="${POINT_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
EXPERIMENT_BASENAME="SatVideoIRSDT_v1__${POINT_STAMP}__PointCenter-F1-${POINT_SLUG}_E${POINT_EPOCHS}"
EXPERIMENT_NAME="$RUN_DATE/$EXPERIMENT_BASENAME"
EXPERIMENT_DIR="$DAY_ROOT/$EXPERIMENT_BASENAME"
STATUS_DIR="$DAY_ROOT/_pointcenter_pipeline_status"
STATUS_FILE="$STATUS_DIR/${POINT_SLUG}.status"

csig_require_allowed_gpus "$GPU_SPEC"
if [[ "$POINT_BATCH_SIZE" -le 0 || $((POINT_BATCH_SIZE % GPU_NUM)) -ne 0 ]]; then
    echo "Global batch must be positive and divisible by three." >&2
    exit 2
fi
if [[ $((POINT_BATCH_SIZE / GPU_NUM)) -gt 7 ]]; then
    echo "PointCenter safety cap is per-rank batch 7 on 24 GiB RTX 3090." >&2
    exit 2
fi
if [[ -z "${SWANLAB_API_KEY:-}" && -r "$SWANLAB_CREDENTIAL_FILE" ]]; then
    SWANLAB_API_KEY="$(awk '$1 == "password" {print $2; exit}' "$SWANLAB_CREDENTIAL_FILE")"
    export SWANLAB_API_KEY
fi
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    echo "SwanLab cloud credential is unavailable." >&2
    exit 1
fi
if [[ "$POINT_RESUME_MODE" == "auto" && -z "$POINT_SWANLAB_ID" ]]; then
    echo "A resumed experiment must provide POINT_SWANLAB_ID." >&2
    exit 2
fi
if [[ -e "$EXPERIMENT_DIR" && "$POINT_RESUME_MODE" != "auto" ]]; then
    echo "Refusing to reuse experiment directory: $EXPERIMENT_DIR" >&2
    exit 1
fi
if [[ "$POINT_RESUME_MODE" == "auto" && ! -f "$EXPERIMENT_DIR/checkpoints/latest_model.pth" ]]; then
    echo "Resume requested but latest checkpoint is unavailable: $EXPERIMENT_DIR" >&2
    exit 1
fi

swanlab_id_arguments=()
if [[ -n "$POINT_SWANLAB_ID" ]]; then
    swanlab_id_arguments=(--swanlab_id "$POINT_SWANLAB_ID")
fi

mkdir -p "$STATUS_DIR"
printf 'RUNNING gpu=%s model=%s scratch=1 global_batch=%s started=%s\n' \
    "$GPU_SPEC" "$MODEL" "$POINT_BATCH_SIZE" \
    "$(date --iso-8601=seconds)" >"$STATUS_FILE"

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
    --learning_rate 0.001 \
    --base_lr_mult 5.0 \
    --decay_rate 0.0001 \
    --batch_size "$POINT_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --epoch "$POINT_EPOCHS" \
    --early_stopping_patience 0 \
    --seqlen 40 \
    --patch_size 128 \
    --sample_rate 0.04 \
    --sequence_augmentation 1 \
    --step_size 10 \
    --lr_decay 0.7 \
    --threshold_eval 0.5 \
    --train_workers 12 \
    --val_workers 2 \
    --prefetch_factor 2 \
    --loss center_consistency_f1 \
    --mask_padded_frames 1 \
    --tversky_fp_weight 0.6 \
    --tversky_fn_weight 0.4 \
    --hard_negative_topk 4096 \
    --f1_ohem_dice_weight 0.15 \
    --f1_ohem_hard_weight 0.10 \
    --f1_ohem_negative_ratio 4.0 \
    --f1_ohem_min_negatives 256 \
    --f1_ohem_margin 1.0 \
    --f1_ohem_warmup_epochs 5 \
    --f1_ohem_ramp_epochs 10 \
    --point_center_weight 0.05 \
    --point_consistency_weight 0.01 \
    --point_consistency_temperature 1.0 \
    --point_center_sigma 1.25 \
    --point_center_fusion_weight 0.25 \
    --structure_variant raw_apmd_hybrid_rms \
    --structure_bottleneck_channels 8 \
    --structure_max_shift 4.0 \
    --eval_chunk_rows "$TEST_EVAL_CHUNK_ROWS" \
    --train_amp 1 \
    --eval_amp 1 \
    --eval_interval "$POINT_EVAL_INTERVAL" \
    --resume "$POINT_RESUME_MODE" \
    --seed "$POINT_SEED" \
    --deterministic 0 \
    --run_test_after_train 0 \
    --base_ckpt '' \
    --spatial_ckpt '' \
    --st_ckpt '' \
    --freeze_pretrained 0 \
    --use_swanlab 1 \
    --swanlab_project CSIG2026-DeepPro \
    --swanlab_group "pointcenter_f1_scratch_${RUN_DATE}" \
    --swanlab_mode cloud \
    --swanlab_resume "$POINT_SWANLAB_RESUME" \
    "${swanlab_id_arguments[@]}"

MODEL_LOG="$EXPERIMENT_DIR/logs/${MODEL}.txt"
for expected in "base_ckpt=''" "spatial_ckpt=''" "st_ckpt=''"; do
    if ! grep -Fq "$expected" "$MODEL_LOG"; then
        echo "Scratch-only audit failed: missing $expected in model log." >&2
        exit 1
    fi
done
if ! grep -Fq 'from random weights; no base checkpoint loaded.' "$MODEL_LOG"; then
    echo "Scratch-only audit failed: random initialization was not logged." >&2
    exit 1
fi

POST_ROOT="$EXPERIMENT_DIR/postprocess"
PROBABILITY_ROOT="$POST_ROOT/probabilities"
RESULT_ROOT="$POST_ROOT/results"
LOG_ROOT="$POST_ROOT/logs"
SUBMISSION_ROOT="$EXPERIMENT_DIR/submission"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
mkdir -p "$PROBABILITY_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$SUBMISSION_ROOT"

mapfile -t CHECKPOINT_SELECTORS < <(
    "$PYTHON_BIN" tools/select_eval_checkpoints.py \
        --log-file "$MODEL_LOG" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --top-k 3 \
        --output-json "$RESULT_ROOT/checkpoint_candidates.json"
)
if [[ "${#CHECKPOINT_SELECTORS[@]}" -gt 3 ]]; then
    CHECKPOINT_SELECTORS=("${CHECKPOINT_SELECTORS[@]:0:3}")
fi
if [[ "${#CHECKPOINT_SELECTORS[@]}" -eq 0 ]]; then
    echo "No evaluated checkpoints were selected." >&2
    exit 1
fi

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

    "$PYTHON_BIN" -u test.py \
        --gpu "$POSTPROCESS_GPU" \
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
        --output-csv "$RESULT_ROOT/${selector_slug}_centroid_f1.csv" \
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
ZIP_PATH="$SUBMISSION_ROOT/submit_${POINT_SLUG}_best_proxy_f1.zip"
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

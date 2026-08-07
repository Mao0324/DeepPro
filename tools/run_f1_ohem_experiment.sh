#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
PYTHON_BIN="/home/devbox/project/model/miniconda3/envs/sjyPID/bin/python"
DATA_ROOT="/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1"
BASE_CHECKPOINT="$REPO_ROOT/pretrained/SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth"

if [[ $# -ne 7 ]]; then
    echo "Usage: $0 MODEL GPU EXPERIMENT LR BASE_LR_MULT OUTPUT_SLUG SWANLAB_MODE" >&2
    exit 2
fi

MODEL="$1"
GPU_ID="$2"
EXPERIMENT_NAME="$3"
LEARNING_RATE="$4"
BASE_LR_MULT="$5"
OUTPUT_SLUG="$6"
SWANLAB_MODE="$7"

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
    --learning_rate "$LEARNING_RATE" \
    --base_lr_mult "$BASE_LR_MULT" \
    --decay_rate 0.0001 \
    --batch_size 20 \
    --epoch 50 \
    --seqlen 40 \
    --patch_size 128 \
    --sample_rate 0.04 \
    --step_size 10 \
    --lr_decay 0.7 \
    --threshold_eval 0.5 \
    --train_workers 8 \
    --val_workers 4 \
    --prefetch_factor 2 \
    --loss f1_calibrated_ohem \
    --tversky_fp_weight 0.6 \
    --tversky_fn_weight 0.4 \
    --hard_negative_topk 4096 \
    --base_ckpt "$BASE_CHECKPOINT" \
    --brtd_use_background 1 \
    --brtd_adaptive_tdc 1 \
    --brtd_use_gate 1 \
    --brtd_zero_init 1 \
    --eval_chunk_rows 64 \
    --resume never \
    --seed 46 \
    --deterministic 0 \
    --run_test_after_train 0 \
    --use_swanlab 1 \
    --swanlab_project CSIG2026-DeepPro \
    --swanlab_group f1_calibrated_ohem_three_models \
    --swanlab_mode "$SWANLAB_MODE"

"$REPO_ROOT/tools/finalize_f1_ohem_submission.sh" \
    "$EXPERIMENT_NAME" "$GPU_ID" "$OUTPUT_SLUG"

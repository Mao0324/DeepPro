#!/usr/bin/env bash
set -euo pipefail

# One-shot finalizer. This script never waits for or monitors training.
REPO_ROOT="/home/devbox/project/model/sjy/CSIG2026/Deeppro_v2/DeepPro-main"
PYTHON_BIN="/home/devbox/project/model/miniconda3/envs/sjyPID/bin/python"
DATA_ROOT="/home/devbox/project/model/sjy/CSIG2026/datasets/SatVideoIRSDT_v1"
EXPERIMENT_NAME="SatVideoIRSDT_v1__2026-08-06_10-46-30__tversky_hard_focal_Pretrained_DeepPro-Plus_BRTD2_DataL40_E50"
EXPERIMENT_DIR="$REPO_ROOT/log/sem_seg/$EXPERIMENT_NAME"
GPU_ID="${1:-0}"

LATEST_CHECKPOINT="$EXPERIMENT_DIR/checkpoints/latest_model.pth"
BEST_CHECKPOINT="$EXPERIMENT_DIR/checkpoints/best_model.pth"
SUBMISSION_DIR="$EXPERIMENT_DIR/submission"
CENTROID_DIR="$SUBMISSION_DIR/centroid_thr0p50"
TRACKED_DIR="$SUBMISSION_DIR/tracked_thr0p50"
ZIP_PATH="$SUBMISSION_DIR/submit_brtd2_tversky_hard_focal_best_thr0p50.zip"

if [[ ! -f "$LATEST_CHECKPOINT" || ! -f "$BEST_CHECKPOINT" ]]; then
    echo "Training checkpoints are not ready: $EXPERIMENT_DIR" >&2
    exit 1
fi

"$PYTHON_BIN" - "$LATEST_CHECKPOINT" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")
epoch = int(checkpoint.get("epoch", -1))
if epoch < 49:
    raise SystemExit(
        "Training is incomplete: latest checkpoint is zero-based epoch %d; "
        "expected at least 49 for a 50-epoch run." % epoch
    )
print("Confirmed completed 50-epoch checkpoint:", path)
PY

mkdir -p "$SUBMISSION_DIR"
if [[ -e "$CENTROID_DIR" || -e "$TRACKED_DIR" || -e "$ZIP_PATH" ]]; then
    echo "Refusing to overwrite an existing output under $SUBMISSION_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" -u test.py \
    --gpu "$GPU_ID" \
    --seqlen 40 \
    --datapath "$DATA_ROOT" \
    --dataset SatVideoIRSDT_v1 \
    --logpath "$REPO_ROOT/log" \
    --log_dir "$EXPERIMENT_NAME" \
    --threshold_eval 0.5 \
    --centroid_txt \
    --centroid_threshold 0.5 \
    --centroid_dir "$CENTROID_DIR" \
    --test_workers 2 \
    --prefetch_factor 1 \
    --eval_chunk_rows 64

"$PYTHON_BIN" -u tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py \
    --input-root "$CENTROID_DIR" \
    --output-dir "$TRACKED_DIR" \
    --max-distance 20 \
    --distance-growth 5 \
    --max-missed 2 \
    --velocity-smoothing 0.5 \
    --zip-output "$ZIP_PATH"

"$PYTHON_BIN" - "$ZIP_PATH" <<'PY'
import math
import sys
import zipfile
from pathlib import PurePosixPath

path = sys.argv[1]
with zipfile.ZipFile(path) as archive:
    names = archive.namelist()
    if len(names) != 255 or len(set(names)) != 255:
        raise SystemExit(
            "Invalid member count: expected 255 unique TXT files, got %d"
            % len(names)
        )
    for name in names:
        member = PurePosixPath(name)
        if len(member.parts) != 1 or member.suffix != ".txt":
            raise SystemExit("Non-flat or non-TXT archive member: %s" % name)
        for line_number, line in enumerate(
            archive.read(name).decode("utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if len(fields) < 2:
                raise SystemExit("Malformed line %s:%d" % (name, line_number))
            int(fields[0])
            count = int(fields[1])
            if count < 0 or len(fields) != 2 + 3 * count:
                raise SystemExit("Malformed count %s:%d" % (name, line_number))
            for index in range(count):
                int(fields[2 + 3 * index])
                x = float(fields[3 + 3 * index])
                y = float(fields[4 + 3 * index])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise SystemExit("Non-finite coordinate %s:%d" % (name, line_number))
print("Validated: 255 flat TXT files with target_id x y fields")
PY

sha256sum "$ZIP_PATH"
echo "Submission ZIP: $ZIP_PATH"

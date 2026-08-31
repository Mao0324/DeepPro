#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$RELEASE_ROOT/../.." && pwd)"

cd "$RELEASE_ROOT"
sha256sum -c SHA256SUMS

FINAL_ZIP="$RELEASE_ROOT/artifacts/submit_hrms_scratch_epoch86_adaptive_thr0p16_highres0p96.zip"
CHECKPOINT="$RELEASE_ROOT/checkpoint/epoch_86_model.pth"

source "$REPO_ROOT/tools/project_runtime_env.sh"
"$PYTHON_BIN" "$REPO_ROOT/tools/validate_submission_zip.py" \
    "$FINAL_ZIP" --data-root "$DATA_ROOT" --split test
unzip -tq "$FINAL_ZIP"

grep -Fq "base_ckpt=''" "$RELEASE_ROOT/evidence/scratch_training_evidence.txt"
grep -Fq "spatial_ckpt=''" "$RELEASE_ROOT/evidence/scratch_training_evidence.txt"
grep -Fq "st_ckpt=''" "$RELEASE_ROOT/evidence/scratch_training_evidence.txt"
grep -Fq 'from random weights; no base checkpoint loaded.' \
    "$RELEASE_ROOT/evidence/scratch_training_evidence.txt"
grep -Fq 'Starting a new experiment from scratch.' \
    "$RELEASE_ROOT/evidence/scratch_training_evidence.txt"

"$PYTHON_BIN" - "$CHECKPOINT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location='cpu')
assert checkpoint['model_name'] == 'DeepPro-Plus_BRTD3'
assert checkpoint['model_config']['structure_variant'] == 'raw_apmd_hybrid_rms'
assert checkpoint['epoch'] == 85  # zero-based epoch stored after epoch 86
print('CHECKPOINT_METADATA=PASS model=DeepPro-Plus_BRTD3 structure=raw_apmd_hybrid_rms epoch=86')
PY

while IFS='|' read -r snapshot repository_file; do
    cmp -s "$RELEASE_ROOT/source_snapshot/$snapshot" "$REPO_ROOT/$repository_file" || {
        echo "Source snapshot differs from branch file: $repository_file" >&2
        exit 1
    }
done <<'EOF'
test.py|test.py
train.py|train.py
TestDataLoader.py|data_utils/TestDataLoader.py
TrainDataLoader.py|data_utils/TrainDataLoader.py
project_runtime_env.sh|tools/project_runtime_env.sh
validate_submission_zip.py|tools/validate_submission_zip.py
seg2tracked_centroid_txt.py|tools_forSatVideoIRSTD/seg2tracked_centroid_txt.py
EOF

echo 'RELEASE_VERIFICATION=PASS score_record=91.30 initialization=scratch_only'

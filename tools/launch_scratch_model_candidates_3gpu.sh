#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"
QUEUE_RUNNER="$REPO_ROOT/tools/run_structure_candidate_queue.sh"

DRY_RUN=0
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    if [[ "$1" != "--dry-run" ]]; then
        echo "Usage: $0 [--dry-run]" >&2
        exit 2
    fi
    DRY_RUN=1
fi

BATCH_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DATE="${BATCH_STAMP%%_*}"
DAY_ROOT="$REPO_ROOT/log/sem_seg/$RUN_DATE"
SCREEN_LOG_ROOT="$DAY_ROOT/_scratch_candidate_screen_logs"
STRUCTURE_SEED="${STRUCTURE_SEED:-47}"
STRUCTURE_USE_SWANLAB="${STRUCTURE_USE_SWANLAB:-1}"
STRUCTURE_SWANLAB_MODE="${STRUCTURE_SWANLAB_MODE:-cloud}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-/home/user/.swanlab/.netrc}"
STRUCTURE_BATCH_SIZE="${STRUCTURE_BATCH_SIZE:-10}"
STRUCTURE_GRAD_ACCUM_STEPS="${STRUCTURE_GRAD_ACCUM_STEPS:-2}"
SWANLAB_GROUP="scratch_model_candidates_seed${STRUCTURE_SEED}_${BATCH_STAMP}"
GPU_IDS=(0 1 2)

# Each candidate is compared with the historical seed-47 Hybrid-RMS scratch
# baseline. Candidate 2 and 3 add one module on top of candidate 1, so their
# effects remain attributable within this parallel batch.
CONFIGURATIONS=(
    "0|raw_apmd_hybrid_rms_scratch_init|scratch|hrms_scratch_init_seed${STRUCTURE_SEED}"
    "1|raw_apmd_hybrid_rms_scratch_bandpass|scratch|hrms_scratch_bandpass_seed${STRUCTURE_SEED}"
    "2|raw_apmd_hybrid_rms_scratch_detail|scratch|hrms_scratch_detail_seed${STRUCTURE_SEED}"
)

if [[ "$STRUCTURE_USE_SWANLAB" != "1" ]]; then
    echo "This launcher requires STRUCTURE_USE_SWANLAB=1." >&2
    exit 2
fi
if [[ "$STRUCTURE_SWANLAB_MODE" != "cloud" ]]; then
    echo "This launcher requires STRUCTURE_SWANLAB_MODE=cloud." >&2
    exit 2
fi
if [[ -z "${SWANLAB_API_KEY:-}" && ! -r "$SWANLAB_CREDENTIAL_FILE" ]]; then
    echo "SwanLab cloud credential is unavailable: $SWANLAB_CREDENTIAL_FILE" >&2
    exit 1
fi

for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu variant init_mode slug <<<"$configuration"
    csig_require_allowed_gpu "$gpu"
    if [[ "$init_mode" != "scratch" ]]; then
        echo "Scratch-only policy rejects initialization: $init_mode" >&2
        exit 2
    fi
    experiment="$DAY_ROOT/SatVideoIRSDT_v1__${BATCH_STAMP}__F1OHEM-${slug}_E100"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "GPU=$gpu init=$init_mode variant=$variant experiment=$experiment"
        echo "  physical_batch=$STRUCTURE_BATCH_SIZE accumulation=$STRUCTURE_GRAD_ACCUM_STEPS effective_batch=$((STRUCTURE_BATCH_SIZE * STRUCTURE_GRAD_ACCUM_STEPS))"
        echo "  expected_zip=$experiment/submission/submit_${slug}_best_proxy_f1.zip"
    elif [[ -e "$experiment" ]]; then
        echo "Refusing to reuse existing experiment: $experiment" >&2
        exit 1
    fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run only: 3 scratch candidates on GPUs $CSIG_ALLOWED_GPU_IDS with SwanLab cloud logging."
    exit 0
fi

mapfile -t GPU_MEMORY < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
if [[ "${#GPU_MEMORY[@]}" -lt 3 ]]; then
    echo "Expected physical GPUs 0-2, found ${#GPU_MEMORY[@]} device(s)." >&2
    exit 1
fi
for gpu in "${GPU_IDS[@]}"; do
    used_memory="${GPU_MEMORY[$gpu]//[[:space:]]/}"
    if [[ ! "$used_memory" =~ ^[0-9]+$ || "$used_memory" -gt 1024 ]]; then
        echo "GPU $gpu is not idle: ${used_memory:-unknown} MiB used" >&2
        exit 1
    fi
done

mkdir -p "$SCREEN_LOG_ROOT" "$DAY_ROOT/_structure_pipeline_status"
for configuration in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r gpu variant init_mode slug <<<"$configuration"
    session="csig_scratch_candidate_g${gpu}_s${STRUCTURE_SEED}_${BATCH_STAMP}"
    screen_log="$SCREEN_LOG_ROOT/${session}.log"
    if screen -S "$session" -Q select . >/dev/null 2>&1; then
        echo "Screen session already exists: $session" >&2
        exit 1
    fi
    if [[ -e "$screen_log" ]]; then
        echo "Screen log already exists: $screen_log" >&2
        exit 1
    fi
    screen -dmS "$session" -L -Logfile "$screen_log" \
        env \
            STRUCTURE_SEED="$STRUCTURE_SEED" \
            STRUCTURE_USE_SWANLAB="$STRUCTURE_USE_SWANLAB" \
            STRUCTURE_SWANLAB_MODE="$STRUCTURE_SWANLAB_MODE" \
            SWANLAB_CREDENTIAL_FILE="$SWANLAB_CREDENTIAL_FILE" \
            STRUCTURE_BATCH_SIZE="$STRUCTURE_BATCH_SIZE" \
            STRUCTURE_GRAD_ACCUM_STEPS="$STRUCTURE_GRAD_ACCUM_STEPS" \
        bash "$QUEUE_RUNNER" \
            "$gpu" "$BATCH_STAMP" "$SWANLAB_GROUP" \
            "$variant|$init_mode|$slug"
    echo "Started GPU $gpu: $session"
    echo "  SwanLab group: $SWANLAB_GROUP"
    echo "  log: $screen_log"
done

echo "Started 3 scratch-only candidates on GPUs 0,1,2."
echo "Each successful pipeline will create, validate, and hash a submission ZIP."

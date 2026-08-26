#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/project_runtime_env.sh"

POSTPROCESS_RUNNER="$REPO_ROOT/tools/resume_structure_candidate_postprocess.sh"
EXPERIMENT_RUNNER="$REPO_ROOT/tools/run_structure_candidate_experiment.sh"
QUEUE_ROOT="$REPO_ROOT/log/sem_seg/2026-08-25/_structure_pipeline_status"
QUEUE_STATUS="$QUEUE_ROOT/init_then_bandpass_ddp3.status"

INIT_VARIANT="raw_apmd_hybrid_rms_scratch_init"
INIT_SLUG="hrms_scratch_init_ddp3_seed47"
INIT_BATCH_STAMP="2026-08-25_09-30-18"
INIT_EXPERIMENT="$REPO_ROOT/log/sem_seg/2026-08-25/SatVideoIRSDT_v1__${INIT_BATCH_STAMP}__F1OHEM-${INIT_SLUG}_E100"
INIT_ZIP="$INIT_EXPERIMENT/submission/submit_${INIT_SLUG}_best_proxy_f1.zip"

NEXT_GPU_SPEC="0,1,2"
NEXT_VARIANT="raw_apmd_hybrid_rms_scratch_bandpass"
NEXT_SEED="47"
NEXT_SLUG="hrms_scratch_bandpass_ddp3_seed${NEXT_SEED}"
SWANLAB_CREDENTIAL_FILE="${SWANLAB_CREDENTIAL_FILE:-/home/user/.swanlab/.netrc}"

mkdir -p "$QUEUE_ROOT"
printf 'POSTPROCESSING slug=%s started=%s\n' \
    "$INIT_SLUG" "$(date --iso-8601=seconds)" >"$QUEUE_STATUS"

on_error() {
    local exit_code=$?
    printf 'FAILED exit=%s time=%s\n' \
        "$exit_code" "$(date --iso-8601=seconds)" >"$QUEUE_STATUS"
    exit "$exit_code"
}
trap on_error ERR

bash "$POSTPROCESS_RUNNER" \
    0 "$INIT_VARIANT" "$INIT_SLUG" "$INIT_BATCH_STAMP"

if [[ ! -s "$INIT_ZIP" || ! -s "$INIT_ZIP.sha256" || ! -f "$INIT_EXPERIMENT/COMPLETE" ]]; then
    echo "Recovered experiment is missing a validated submission artifact: $INIT_ZIP" >&2
    exit 1
fi

mapfile -t GPU_MEMORY < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
if [[ "${#GPU_MEMORY[@]}" -lt 3 ]]; then
    echo "Expected physical GPUs 0-2, found ${#GPU_MEMORY[@]} device(s)." >&2
    exit 1
fi
for gpu in 0 1 2; do
    used_memory="${GPU_MEMORY[$gpu]//[[:space:]]/}"
    if [[ ! "$used_memory" =~ ^[0-9]+$ || "$used_memory" -gt 1024 ]]; then
        echo "GPU $gpu is not idle after postprocessing: ${used_memory:-unknown} MiB used" >&2
        exit 1
    fi
done

NEXT_BATCH_STAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
NEXT_RUN_DATE="${NEXT_BATCH_STAMP%%_*}"
NEXT_EXPERIMENT="$REPO_ROOT/log/sem_seg/$NEXT_RUN_DATE/SatVideoIRSDT_v1__${NEXT_BATCH_STAMP}__F1OHEM-${NEXT_SLUG}_E100"
NEXT_SWANLAB_GROUP="priority_scratch_bandpass_ddp3_seed${NEXT_SEED}_${NEXT_BATCH_STAMP}"

if [[ -e "$NEXT_EXPERIMENT" ]]; then
    echo "Refusing to reuse existing experiment: $NEXT_EXPERIMENT" >&2
    exit 1
fi

printf 'TRAINING slug=%s experiment=%s started=%s\n' \
    "$NEXT_SLUG" "$NEXT_EXPERIMENT" "$(date --iso-8601=seconds)" >"$QUEUE_STATUS"

export STRUCTURE_ADAPTER_LR=0.001
export STRUCTURE_BASE_LR_MULT=5.0
export STRUCTURE_SEED="$NEXT_SEED"
export STRUCTURE_BATCH_SIZE=18
export STRUCTURE_GRAD_ACCUM_STEPS=1
export STRUCTURE_EVAL_INTERVAL=5
export STRUCTURE_INIT_MODE=scratch
export STRUCTURE_RESUME_MODE=never
export STRUCTURE_USE_SWANLAB=1
export STRUCTURE_SWANLAB_MODE=cloud
export STRUCTURE_SWANLAB_RESUME=never
export SWANLAB_CREDENTIAL_FILE
export THRESHOLD_GRID="${THRESHOLD_GRID:-0.10:0.95:0.01}"

bash "$EXPERIMENT_RUNNER" \
    "$NEXT_GPU_SPEC" "$NEXT_VARIANT" "$NEXT_SLUG" \
    "$NEXT_BATCH_STAMP" "$NEXT_SWANLAB_GROUP"

printf 'COMPLETE slug=%s experiment=%s time=%s\n' \
    "$NEXT_SLUG" "$NEXT_EXPERIMENT" "$(date --iso-8601=seconds)" >"$QUEUE_STATUS"


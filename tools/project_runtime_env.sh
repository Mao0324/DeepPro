#!/usr/bin/env bash

# Resolve migration-sensitive paths once for all launchers. Callers may override
# any value through the environment before sourcing this file.
PROJECT_RUNTIME_TOOLS_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd
)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$PROJECT_RUNTIME_TOOLS_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/../datasets/SatVideoIRSDT_v1}"
TEST_USE_AMP="${TEST_USE_AMP:-1}"
TEST_EVAL_CHUNK_ROWS="${TEST_EVAL_CHUNK_ROWS:-32}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "sjyPID" ]]; then
        PYTHON_BIN="$(command -v python || true)"
    fi

    if [[ -z "${PYTHON_BIN:-}" ]]; then
        PROJECT_RUNTIME_CONDA_BIN="$(command -v conda || true)"
        if [[ -n "$PROJECT_RUNTIME_CONDA_BIN" ]]; then
            PROJECT_RUNTIME_CONDA_BASE="$(conda info --base 2>/dev/null || true)"
            PROJECT_RUNTIME_CONDA_PYTHON="$PROJECT_RUNTIME_CONDA_BASE/envs/sjyPID/bin/python"
            if [[ -x "$PROJECT_RUNTIME_CONDA_PYTHON" ]]; then
                PYTHON_BIN="$PROJECT_RUNTIME_CONDA_PYTHON"
            fi
        fi
    fi
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
    echo "Unable to locate the sjyPID Python. Activate sjyPID or set PYTHON_BIN." >&2
    return 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Dataset directory is unavailable: $DATA_ROOT" >&2
    echo "Set DATA_ROOT to the migrated SatVideoIRSDT_v1 directory." >&2
    return 1
fi
if [[ "$TEST_USE_AMP" != "0" && "$TEST_USE_AMP" != "1" ]]; then
    echo "TEST_USE_AMP must be 0 or 1, got: $TEST_USE_AMP" >&2
    return 1
fi
if [[ ! "$TEST_EVAL_CHUNK_ROWS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TEST_EVAL_CHUNK_ROWS must be a positive integer, got: $TEST_EVAL_CHUNK_ROWS" >&2
    return 1
fi

export REPO_ROOT PYTHON_BIN DATA_ROOT TEST_USE_AMP TEST_EVAL_CHUNK_ROWS
export PYTORCH_CUDA_ALLOC_CONF

# New-server safety policy: GPU 3 is excluded and no job may target a device
# outside physical GPUs 0, 1, and 2.
CSIG_ALLOWED_GPU_IDS="0,1,2"
export CSIG_ALLOWED_GPU_IDS

csig_require_allowed_gpu() {
    local gpu_id="$1"
    local allowed=",${CSIG_ALLOWED_GPU_IDS//[[:space:]]/},"
    if [[ "$allowed" != *",$gpu_id,"* ]]; then
        echo "GPU $gpu_id is blocked by the new-server policy; allowed GPUs: $CSIG_ALLOWED_GPU_IDS" >&2
        return 1
    fi
}

csig_require_allowed_gpus() {
    local gpu_spec="$1"
    local gpu_ids=()
    local gpu_id
    IFS=',' read -r -a gpu_ids <<<"$gpu_spec"
    if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
        echo "At least one GPU must be specified." >&2
        return 1
    fi
    for gpu_id in "${gpu_ids[@]}"; do
        gpu_id="${gpu_id//[[:space:]]/}"
        if [[ -z "$gpu_id" ]]; then
            echo "Invalid GPU list: $gpu_spec" >&2
            return 1
        fi
        csig_require_allowed_gpu "$gpu_id"
    done
}

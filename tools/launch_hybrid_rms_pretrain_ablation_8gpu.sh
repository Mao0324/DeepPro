#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "The pretrained/8-GPU launcher is retired by the scratch-only and GPU 0/1/2 policies." >&2
echo "Redirecting to the scratch-only three-GPU launcher." >&2
exec bash "$SCRIPT_DIR/launch_hybrid_rms_scratch_3gpu.sh" "$@"

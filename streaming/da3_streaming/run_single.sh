#!/bin/bash
# =============================================================================
# Run a SINGLE pipeline (IRLS or TurboReg) independently for fair timing.
#
# Usage:
#   bash run_single.sh <method> <image_dir> [config]
#
# Examples:
#   bash run_single.sh irls /path/to/kitti/sequences/05/image_2
#   bash run_single.sh turboreg /path/to/kitti/sequences/05/image_2
#   bash run_single.sh turboreg /path/to/tum/rgbd_dataset_freiburg1_xyz/rgb configs/tum_turboreg.yaml
# =============================================================================

set -e

export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

METHOD="${1:?Usage: bash run_single.sh <irls|turboreg> <image_dir> [config]}"
IMAGE_DIR="${2:?Usage: bash run_single.sh <irls|turboreg> <image_dir> [config]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$METHOD" = "irls" ]; then
    CONFIG="${3:-configs/base_config.yaml}"
    echo "[Running DA3 + IRLS] config=$CONFIG"
    python da3_streaming.py --image_dir "$IMAGE_DIR" --config "$CONFIG"
elif [ "$METHOD" = "turboreg" ]; then
    CONFIG="${3:-configs/turboreg_config.yaml}"
    echo "[Running DA3 + TurboReg] config=$CONFIG"
    python da3_streaming_turboreg.py --image_dir "$IMAGE_DIR" --config "$CONFIG"
else
    echo "ERROR: method must be 'irls' or 'turboreg'"
    exit 1
fi

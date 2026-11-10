#!/bin/bash
# =============================================================================
# DA3-Streaming: Run IRLS and TurboReg SEPARATELY to avoid OOM.
# Each method runs in its own process, GPU fully freed between runs.
#
# Usage:
#   bash run_separate.sh <image_dir> <gt_poses_file> [gt_format]
#
# Example (KITTI):
#   CUDA_VISIBLE_DEVICES=2 bash run_separate.sh \
#       /path/to/kitti/sequences/05/image_2 \
#       /path/to/kitti/poses/05.txt kitti
#
# Example (TUM):
#   CUDA_VISIBLE_DEVICES=2 bash run_separate.sh \
#       /path/to/tum/rgbd_dataset_freiburg1_xyz/rgb \
#       /path/to/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt tum
# =============================================================================

set -e

export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

IMAGE_DIR="${1:?Usage: bash run_separate.sh <image_dir> <gt_poses_file> [gt_format]}"
GT_POSES="${2:?Usage: bash run_separate.sh <image_dir> <gt_poses_file> [gt_format]}"
GT_FORMAT="${3:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-select config based on gt_format
if [ "$GT_FORMAT" = "tum" ]; then
    IRLS_CONFIG="configs/tum.yaml"
    TR_CONFIG="configs/tum_turboreg.yaml"
else
    IRLS_CONFIG="configs/base_config.yaml"
    TR_CONFIG="configs/turboreg_config.yaml"
fi

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SANITIZED_DIR=$(echo "$IMAGE_DIR" | tr '/' '_')
EVAL_OUTPUT="/path/to/eval_results/da3_${SANITIZED_DIR}_${TIMESTAMP}"

echo "============================================================"
echo "DA3-Streaming: IRLS vs TurboReg (Separate Runs)"
echo "============================================================"
echo "Image dir:    $IMAGE_DIR"
echo "GT poses:     $GT_POSES"
echo "GT format:    $GT_FORMAT"
echo "IRLS config:  $IRLS_CONFIG"
echo "TR config:    $TR_CONFIG"
echo "============================================================"

# ---- Step 1: Run IRLS ----
echo ""
echo "[Step 1/3] Running DA3 + IRLS..."
echo "------------------------------------------------------------"
python da3_streaming.py --image_dir "$IMAGE_DIR" --config "$IRLS_CONFIG"

IRLS_DIR=$(ls -td /path/to/exps/${SANITIZED_DIR}/20* 2>/dev/null | head -1)
echo "IRLS output: $IRLS_DIR"

# GPU is fully freed after python process exits
echo "[GPU freed] Waiting 5s for GPU cooldown..."
sleep 5

# ---- Step 2: Run TurboReg ----
echo ""
echo "[Step 2/3] Running DA3 + TurboReg..."
echo "------------------------------------------------------------"
python da3_streaming_turboreg.py --image_dir "$IMAGE_DIR" --config "$TR_CONFIG"

TURBOREG_DIR=$(ls -td /path/to/exps/${SANITIZED_DIR}_turboreg/20* 2>/dev/null | head -1)
echo "TurboReg output: $TURBOREG_DIR"

# ---- Step 3: Evaluate ----
echo ""
echo "[Step 3/3] Evaluating..."
echo "------------------------------------------------------------"
python eval_compare.py \
    --gt_poses "$GT_POSES" \
    --gt_format "$GT_FORMAT" \
    --irls_dir "$IRLS_DIR" \
    --turboreg_dir "$TURBOREG_DIR" \
    --output_dir "$EVAL_OUTPUT"

echo ""
echo "============================================================"
echo "DONE! Results: $EVAL_OUTPUT"
echo "============================================================"

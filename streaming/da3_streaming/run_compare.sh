#!/bin/bash
# =============================================================================
# DA3-Streaming: Run both IRLS and TurboReg pipelines, then compare.
#
# Usage:
#   bash run_compare.sh <image_dir> <gt_poses_file> [gt_format]
#
# Example (KITTI):
#   bash run_compare.sh /path/to/kitti/sequences/04/image_2 /path/to/kitti/poses/04.txt kitti
#
# Arguments:
#   image_dir    : Directory containing input images
#   gt_poses_file: Ground truth poses (KITTI or VGGT-Long format)
#   gt_format    : 'kitti' or 'vggtlong' or 'auto' (default: auto)
# =============================================================================

set -e

# Ensure PyTorch libs are found for TurboReg .so
export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

IMAGE_DIR="${1:?Usage: bash run_compare.sh <image_dir> <gt_poses_file> [gt_format]}"
GT_POSES="${2:?Usage: bash run_compare.sh <image_dir> <gt_poses_file> [gt_format]}"
GT_FORMAT="${3:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SANITIZED_DIR=$(echo "$IMAGE_DIR" | tr '/' '_')
EVAL_OUTPUT="/path/to/eval_results/da3_${SANITIZED_DIR}_${TIMESTAMP}"

echo "============================================================"
echo "DA3-Streaming: IRLS vs TurboReg Comparison"
echo "============================================================"
echo "Image dir:   $IMAGE_DIR"
echo "GT poses:    $GT_POSES"
echo "GT format:   $GT_FORMAT"
echo "Eval output: $EVAL_OUTPUT"
echo "============================================================"

# ---- Step 1: Run DA3 + IRLS pipeline ----
echo ""
echo "[Step 1/3] Running DA3 + IRLS pipeline..."
echo "------------------------------------------------------------"
python da3_streaming.py --image_dir "$IMAGE_DIR" --config configs/base_config.yaml 2>&1 | tee /tmp/da3_irls_run.log

# Find the latest IRLS experiment directory
IRLS_DIR=$(ls -td /path/to/exps/${SANITIZED_DIR}/20* 2>/dev/null | head -1)
if [ -z "$IRLS_DIR" ]; then
    echo "ERROR: DA3 IRLS experiment directory not found"
    exit 1
fi
echo "DA3 IRLS output: $IRLS_DIR"

# ---- Step 2: Run DA3 + TurboReg pipeline ----
echo ""
echo "[Step 2/3] Running DA3 + TurboReg pipeline..."
echo "------------------------------------------------------------"
python da3_streaming_turboreg.py --image_dir "$IMAGE_DIR" --config configs/turboreg_config.yaml 2>&1 | tee /tmp/da3_turboreg_run.log

# Find the latest TurboReg experiment directory
TURBOREG_DIR=$(ls -td /path/to/exps/${SANITIZED_DIR}_turboreg/20* 2>/dev/null | head -1)
if [ -z "$TURBOREG_DIR" ]; then
    echo "ERROR: DA3 TurboReg experiment directory not found"
    exit 1
fi
echo "DA3 TurboReg output: $TURBOREG_DIR"

# ---- Step 3: Evaluate and compare ----
echo ""
echo "[Step 3/3] Evaluating and comparing..."
echo "------------------------------------------------------------"
python eval_compare.py \
    --gt_poses "$GT_POSES" \
    --gt_format "$GT_FORMAT" \
    --irls_dir "$IRLS_DIR" \
    --turboreg_dir "$TURBOREG_DIR" \
    --output_dir "$EVAL_OUTPUT"

echo ""
echo "============================================================"
echo "DONE!"
echo "DA3 IRLS results:    $IRLS_DIR"
echo "DA3 TurboReg results: $TURBOREG_DIR"
echo "Evaluation:          $EVAL_OUTPUT"
echo "============================================================"

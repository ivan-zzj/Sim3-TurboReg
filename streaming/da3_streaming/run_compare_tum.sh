#!/bin/bash
# =============================================================================
# DA3-Streaming on TUM RGB-D: Run both IRLS and TurboReg, then compare.
#
# Usage:
#   bash run_compare_tum.sh <image_dir> <gt_poses_file>
#
# Example:
#   bash run_compare_tum.sh /path/to/tum/rgbd_dataset_freiburg1_desk/rgb \
#                           /path/to/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt
#
# Arguments:
#   image_dir    : Directory containing RGB images (e.g. .../rgb/)
#   gt_poses_file: TUM groundtruth.txt (timestamp tx ty tz qx qy qz qw)
# =============================================================================

set -e

IMAGE_DIR="${1:?Usage: bash run_compare_tum.sh <image_dir> <gt_poses_file>}"
GT_POSES="${2:?Usage: bash run_compare_tum.sh <image_dir> <gt_poses_file>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SANITIZED_DIR=$(echo "$IMAGE_DIR" | tr '/' '_')
EVAL_OUTPUT="/path/to/eval_results/da3_tum_${SANITIZED_DIR}_${TIMESTAMP}"

# Ensure PyTorch libs are found for TurboReg .so
export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

echo "============================================================"
echo "DA3-Streaming TUM: IRLS vs TurboReg Comparison"
echo "============================================================"
echo "Image dir:   $IMAGE_DIR"
echo "GT poses:    $GT_POSES"
echo "Eval output: $EVAL_OUTPUT"
echo "============================================================"

# ---- Step 1: Run DA3 + IRLS pipeline ----
echo ""
echo "[Step 1/3] Running DA3 + IRLS (TUM config)..."
echo "------------------------------------------------------------"
python da3_streaming.py --image_dir "$IMAGE_DIR" --config configs/tum.yaml 2>&1 | tee /tmp/da3_tum_irls_run.log

IRLS_DIR=$(ls -td /path/to/exps/${SANITIZED_DIR}/20* 2>/dev/null | head -1)
if [ -z "$IRLS_DIR" ]; then
    echo "ERROR: DA3 IRLS experiment directory not found"
    exit 1
fi
echo "DA3 IRLS output: $IRLS_DIR"

# ---- Step 2: Run DA3 + TurboReg pipeline ----
echo ""
echo "[Step 2/3] Running DA3 + TurboReg (TUM config)..."
echo "------------------------------------------------------------"
python da3_streaming_turboreg.py --image_dir "$IMAGE_DIR" --config configs/tum_turboreg.yaml 2>&1 | tee /tmp/da3_tum_turboreg_run.log

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
    --gt_format tum \
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

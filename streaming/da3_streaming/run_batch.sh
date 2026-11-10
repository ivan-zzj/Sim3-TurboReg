#!/bin/bash
# =============================================================================
# DA3-Streaming: Run a SINGLE method on all KITTI/TUM sequences.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh <method> <dataset> [config]
#
# Examples:
#   # KITTI - IRLS baseline (only need to run once)
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh irls kitti
#
#   # KITTI - TurboReg (run multiple times for ablation)
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh turboreg kitti
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh turboreg kitti configs/turboreg_ablation1.yaml
#   KITTI_SEQS=05 TURBOREG_USE_DISTANCE_DIFF_CONSISTENCY=true \
#     TURBOREG_DISTANCE_DIFF_THRESH=0.03 CUDA_VISIBLE_DEVICES=2 bash run_batch.sh turboreg kitti
#
#   # TUM
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh irls tum
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh turboreg tum
# =============================================================================

set -e

export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

METHOD="${1:?Usage: bash run_batch.sh <irls|turboreg> <kitti|tum> [config]}"
DATASET="${2:?Usage: bash run_batch.sh <irls|turboreg> <kitti|tum> [config]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Default configs
if [ "$METHOD" = "irls" ]; then
    if [ "$DATASET" = "tum" ]; then
        CONFIG="${3:-configs/tum.yaml}"
    else
        CONFIG="${3:-configs/base_config.yaml}"
    fi
    SCRIPT="da3_streaming.py"
elif [ "$METHOD" = "turboreg" ]; then
    if [ "$DATASET" = "tum" ]; then
        CONFIG="${3:-configs/tum_turboreg.yaml}"
    else
        CONFIG="${3:-configs/turboreg_config.yaml}"
    fi
    SCRIPT="da3_streaming_turboreg.py"
else
    echo "ERROR: method must be 'irls' or 'turboreg'"
    exit 1
fi

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SUMMARY_CSV="/path/to/eval_results/da3_${DATASET}_${METHOD}_${TIMESTAMP}.csv"
mkdir -p /path/to/eval_results

echo "seq,ate_rmse,align_time,total_time" > "$SUMMARY_CSV"

echo "============================================================"
echo "DA3-Streaming batch: method=$METHOD  dataset=$DATASET"
echo "Config: $CONFIG"
echo "Python: $PYTHON_BIN"
if [ "$METHOD" = "turboreg" ]; then
    echo "Distance-diff consistency: ${TURBOREG_USE_DISTANCE_DIFF_CONSISTENCY:-config/default}"
    echo "Distance-diff threshold: ${TURBOREG_DISTANCE_DIFF_THRESH:-config/default}"
fi
echo "Summary: $SUMMARY_CSV"
echo "============================================================"

# ---- Define sequences ----
if [ "$DATASET" = "kitti" ]; then
    if [ -n "${KITTI_SEQS:-}" ]; then
        read -r -a SEQS <<< "$KITTI_SEQS"
    else
        SEQS=(00 01 02 03 04 05 06 07 08 09 10)
    fi
    for seq in "${SEQS[@]}"; do
        IMAGE_DIR="/path/to/kitti/sequences/${seq}/image_2"
        GT_FILE="/path/to/kitti/poses/${seq}.txt"
        GT_FORMAT="kitti"
        SEQ_NAME="$seq"

        if [ ! -d "$IMAGE_DIR" ] || [ ! -f "$GT_FILE" ]; then
            echo "WARNING: KITTI $seq not found, skipping..."
            echo "${SEQ_NAME},N/A,N/A,N/A" >> "$SUMMARY_CSV"
            continue
        fi

        echo ""
        echo "===== KITTI $seq ====="
        "$PYTHON_BIN" "$SCRIPT" --image_dir "$IMAGE_DIR" --config "$CONFIG" || { echo "${SEQ_NAME},FAIL,FAIL,FAIL" >> "$SUMMARY_CSV"; continue; }

        # Find output dir and extract timing
        SANITIZED=$(echo "$IMAGE_DIR" | tr '/' '_')
        if [ "$METHOD" = "turboreg" ]; then
            EXP_DIR=$(ls -td /path/to/exps/${SANITIZED}_turboreg/20* 2>/dev/null | head -1)
        else
            EXP_DIR=$(ls -td /path/to/exps/${SANITIZED}/20* 2>/dev/null | head -1)
        fi

        # Eval
        EVAL_DIR="/path/to/eval_results/da3_${METHOD}_${SEQ_NAME}_${TIMESTAMP}"
        "$PYTHON_BIN" eval_compare.py \
            --gt_poses "$GT_FILE" --gt_format "$GT_FORMAT" \
            --${METHOD}_dir "$EXP_DIR" \
            --output_dir "$EVAL_DIR" 2>/dev/null || true

        # Extract to CSV
        EVAL_JSON="${EVAL_DIR}/eval_results.json"
        if [ -f "$EVAL_JSON" ]; then
            "$PYTHON_BIN" -c "
import json
with open('$EVAL_JSON') as f:
    d = json.load(f)
m = d.get('$METHOD', d.get('irls', d.get('turboreg', {})))
ate = m.get('ate_rmse', 'N/A')
at = m.get('alignment_time', 'N/A')
tt = m.get('total_time', 'N/A')
ate_s = f'{ate:.4f}' if isinstance(ate, float) else ate
at_s = f'{at:.2f}' if isinstance(at, float) else at
tt_s = f'{tt:.2f}' if isinstance(tt, float) else tt
print(f'${SEQ_NAME},{ate_s},{at_s},{tt_s}')
" >> "$SUMMARY_CSV"
        else
            echo "${SEQ_NAME},N/A,N/A,N/A" >> "$SUMMARY_CSV"
        fi

        echo "KITTI $seq done"
    done

elif [ "$DATASET" = "tum" ]; then
    TUM_ROOT="/path/to/tum"
    SEQUENCES=(
        "rgbd_dataset_freiburg1_360:360"
        "rgbd_dataset_freiburg1_desk:desk"
        "rgbd_dataset_freiburg1_desk2:desk2"
        "rgbd_dataset_freiburg1_floor:floor"
        "rgbd_dataset_freiburg1_plant:plant"
        "rgbd_dataset_freiburg1_room:room"
        "rgbd_dataset_freiburg1_rpy:rpy"
        "rgbd_dataset_freiburg1_teddy:teddy"
        "rgbd_dataset_freiburg1_xyz:xyz"
    )

    for entry in "${SEQUENCES[@]}"; do
        seq="${entry%%:*}"
        name="${entry##*:}"

        # Optional: ONLY_SEQS="360 plant" limits the batch to those sequences (resume use).
        if [ -n "${ONLY_SEQS:-}" ] && ! echo " $ONLY_SEQS " | grep -q " $name "; then
            continue
        fi

        IMAGE_DIR="${TUM_ROOT}/${seq}/rgb"
        GT_FILE="${TUM_ROOT}/${seq}/groundtruth.txt"
        GT_FORMAT="tum"

        if [ ! -d "$IMAGE_DIR" ] || [ ! -f "$GT_FILE" ]; then
            echo "WARNING: TUM $name not found, skipping..."
            echo "${name},N/A,N/A,N/A" >> "$SUMMARY_CSV"
            continue
        fi

        echo ""
        echo "===== TUM $name ====="
        "$PYTHON_BIN" "$SCRIPT" --image_dir "$IMAGE_DIR" --config "$CONFIG" || { echo "${name},FAIL,FAIL,FAIL" >> "$SUMMARY_CSV"; continue; }

        SANITIZED=$(echo "$IMAGE_DIR" | tr '/' '_')
        if [ "$METHOD" = "turboreg" ]; then
            EXP_DIR=$(ls -td /path/to/exps/${SANITIZED}_turboreg/20* 2>/dev/null | head -1)
        else
            EXP_DIR=$(ls -td /path/to/exps/${SANITIZED}/20* 2>/dev/null | head -1)
        fi

        EVAL_DIR="/path/to/eval_results/da3_${METHOD}_${name}_${TIMESTAMP}"
        "$PYTHON_BIN" eval_compare.py \
            --gt_poses "$GT_FILE" --gt_format "$GT_FORMAT" \
            --${METHOD}_dir "$EXP_DIR" \
            --output_dir "$EVAL_DIR" 2>/dev/null || true

        EVAL_JSON="${EVAL_DIR}/eval_results.json"
        if [ -f "$EVAL_JSON" ]; then
            "$PYTHON_BIN" -c "
import json
with open('$EVAL_JSON') as f:
    d = json.load(f)
m = d.get('$METHOD', d.get('irls', d.get('turboreg', {})))
ate = m.get('ate_rmse', 'N/A')
at = m.get('alignment_time', 'N/A')
tt = m.get('total_time', 'N/A')
ate_s = f'{ate:.4f}' if isinstance(ate, float) else ate
at_s = f'{at:.2f}' if isinstance(at, float) else at
tt_s = f'{tt:.2f}' if isinstance(tt, float) else tt
print(f'${name},{ate_s},{at_s},{tt_s}')
" >> "$SUMMARY_CSV"
        else
            echo "${name},N/A,N/A,N/A" >> "$SUMMARY_CSV"
        fi

        echo "TUM $name done"
    done
else
    echo "ERROR: dataset must be 'kitti' or 'tum'"
    exit 1
fi

# ---- Print summary ----
echo ""
echo "============================================================"
echo "COMPLETE: $METHOD on $DATASET"
echo "Summary: $SUMMARY_CSV"
echo "============================================================"
echo ""
printf "%-8s  %12s  %12s  %12s\n" "Seq" "ATE_RMSE" "Align(s)" "Total(s)"
printf "%s\n" "------------------------------------------------"
tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r s a at tt; do
    printf "%-8s  %12s  %12s  %12s\n" "$s" "$a" "$at" "$tt"
done
echo ""
"$PYTHON_BIN" -c "
import csv
with open('$SUMMARY_CSV') as f:
    rows = [r for r in csv.DictReader(f)]
def avg(key):
    vals = [float(r[key]) for r in rows if r[key] not in ('N/A','FAIL')]
    return sum(vals)/len(vals) if vals else float('nan')
print(f'AVG  ATE={avg(\"ate_rmse\"):.4f}  Align={avg(\"align_time\"):.1f}s  Total={avg(\"total_time\"):.1f}s')
"

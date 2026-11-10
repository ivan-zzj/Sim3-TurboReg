#!/bin/bash
# =============================================================================
# VGGT-Long / Pi3-Long: Run a SINGLE method on all TUM RGB-D freiburg1 seqs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash run_batch_tum.sh <vggt|pi3> <irls|turboreg> [config]
#
# Note: VGGT-Long's own eval_compare.py does not support TUM ground truth;
# evaluation uses DA3-Streaming's eval_compare.py (format-compatible: it reads
# camera_poses.txt / timing.json from the exp dir).
# =============================================================================

set -e

export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

MODEL="${1:?Usage: bash run_batch_tum.sh <vggt|pi3> <irls|turboreg> [config]}"
METHOD="${2:?Usage: bash run_batch_tum.sh <vggt|pi3> <irls|turboreg> [config]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DA3_EVAL="${DA3_EVAL:-$SCRIPT_DIR/../da3_streaming/eval_compare.py}"

if [ "$METHOD" = "irls" ]; then
    SCRIPT="vggt_long.py"
    if [ "$MODEL" = "pi3" ]; then
        CONFIG="${3:-configs/pi3_base_config.yaml}"
    else
        CONFIG="${3:-configs/base_config.yaml}"
    fi
elif [ "$METHOD" = "turboreg" ]; then
    SCRIPT="vggt_long_turboreg.py"
    if [ "$MODEL" = "pi3" ]; then
        CONFIG="${3:-configs/pi3_turboreg_config.yaml}"
    else
        CONFIG="${3:-configs/turboreg_config.yaml}"
    fi
else
    echo "ERROR: method must be 'irls' or 'turboreg'"
    exit 1
fi

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SUMMARY_CSV="/path/to/eval_results/${MODEL}_tum_${METHOD}_${TIMESTAMP}.csv"
mkdir -p /path/to/eval_results

echo "seq,ate_rmse,align_time,total_time" > "$SUMMARY_CSV"

echo "============================================================"
echo "${MODEL}-Long batch: method=$METHOD  dataset=TUM"
echo "Config: $CONFIG"
echo "Summary: $SUMMARY_CSV"
echo "============================================================"

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

    # Optional: ONLY_SEQS="room rpy" limits the batch to those sequences (resume use).
    if [ -n "${ONLY_SEQS:-}" ] && ! echo " $ONLY_SEQS " | grep -q " $name "; then
        continue
    fi
    IMAGE_DIR="${TUM_ROOT}/${seq}/rgb"
    GT_FILE="${TUM_ROOT}/${seq}/groundtruth.txt"

    if [ ! -d "$IMAGE_DIR" ] || [ ! -f "$GT_FILE" ]; then
        echo "WARNING: TUM $name not found, skipping..."
        echo "${name},N/A,N/A,N/A" >> "$SUMMARY_CSV"
        continue
    fi

    echo ""
    echo "===== TUM $name ====="
    python "$SCRIPT" --image_dir "$IMAGE_DIR" --config "$CONFIG" || {
        echo "${name},FAIL,FAIL,FAIL" >> "$SUMMARY_CSV"
        echo "TUM $name FAILED, continuing..."
        continue
    }

    SANITIZED=$(echo "$IMAGE_DIR" | tr '/' '_')
    if [ "$METHOD" = "turboreg" ]; then
        EXP_DIR=$(ls -td /path/to/exps/${SANITIZED}_turboreg/20* 2>/dev/null | head -1)
    else
        EXP_DIR=$(ls -td /path/to/exps/${SANITIZED}/20* 2>/dev/null | head -1)
    fi

    EVAL_DIR="/path/to/eval_results/${MODEL}_${METHOD}_tum_${name}_${TIMESTAMP}"
    python "$DA3_EVAL" \
        --gt_poses "$GT_FILE" --gt_format tum \
        --${METHOD}_dir "$EXP_DIR" \
        --output_dir "$EVAL_DIR" 2>/dev/null || true

    EVAL_JSON="${EVAL_DIR}/eval_results.json"
    if [ -f "$EVAL_JSON" ]; then
        python3 -c "
import json
with open('$EVAL_JSON') as f:
    d = json.load(f)
m = d.get('$METHOD', d.get('irls', d.get('turboreg', {})))
def fmt(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) else 'N/A'
print(f'${name},{fmt(m.get(\"ate_rmse\"))},{fmt(m.get(\"alignment_time\"))},{fmt(m.get(\"total_time\"))}')
" >> "$SUMMARY_CSV"
    else
        echo "${name},N/A,N/A,N/A" >> "$SUMMARY_CSV"
    fi

    echo "TUM $name done"
done

echo ""
echo "============================================================"
echo "COMPLETE: ${MODEL}-Long $METHOD on TUM"
echo "Summary: $SUMMARY_CSV"
echo "============================================================"
echo ""
printf "%-8s  %12s  %12s  %12s\n" "Seq" "ATE_RMSE" "Align(s)" "Total(s)"
printf "%s\n" "------------------------------------------------"
tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r s a at tt; do
    printf "%-8s  %12s  %12s  %12s\n" "$s" "$a" "$at" "$tt"
done
echo ""
python3 -c "
import csv
with open('$SUMMARY_CSV') as f:
    rows = [r for r in csv.DictReader(f)]
def avg(key):
    vals = [float(r[key]) for r in rows if r[key] not in ('N/A','FAIL')]
    return sum(vals)/len(vals) if vals else float('nan')
print(f'AVG  ATE={avg(\"ate_rmse\"):.4f}  Align={avg(\"align_time\"):.1f}s  Total={avg(\"total_time\"):.1f}s')
"

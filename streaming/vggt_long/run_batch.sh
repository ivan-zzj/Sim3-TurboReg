#!/bin/bash
# =============================================================================
# VGGT-Long / Pi3-Long: Run a SINGLE method on all KITTI sequences.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh <model> <method> [config]
#
# Examples:
#   # VGGT-Long IRLS baseline (run once)
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh vggt irls
#
#   # VGGT-Long TurboReg (run for ablation)
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh vggt turboreg
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh vggt turboreg configs/turboreg_ablation1.yaml
#
#   # Pi3-Long IRLS baseline
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh pi3 irls
#
#   # Pi3-Long TurboReg
#   CUDA_VISIBLE_DEVICES=2 bash run_batch.sh pi3 turboreg
# =============================================================================

set -e

export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

MODEL="${1:?Usage: bash run_batch.sh <vggt|pi3> <irls|turboreg> [config]}"
METHOD="${2:?Usage: bash run_batch.sh <vggt|pi3> <irls|turboreg> [config]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Select script and default config
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
SUMMARY_CSV="/path/to/eval_results/${MODEL}_kitti_${METHOD}_${TIMESTAMP}.csv"
mkdir -p /path/to/eval_results

echo "seq,ate_rmse,align_time,total_time" > "$SUMMARY_CSV"

echo "============================================================"
echo "${MODEL}-Long batch: method=$METHOD"
echo "Config: $CONFIG"
echo "Summary: $SUMMARY_CSV"
echo "============================================================"

# Optional: ONLY_SEQS="05 06" limits the batch to those sequences (resume use).
SEQ_LIST="${ONLY_SEQS:-00 01 02 03 04 05 06 07 08 09 10}"
for seq in $SEQ_LIST; do
    IMAGE_DIR="/path/to/kitti/sequences/${seq}/image_2"
    GT_FILE="/path/to/kitti/poses/${seq}.txt"

    if [ ! -d "$IMAGE_DIR" ] || [ ! -f "$GT_FILE" ]; then
        echo "WARNING: KITTI $seq not found, skipping..."
        echo "${seq},N/A,N/A,N/A" >> "$SUMMARY_CSV"
        continue
    fi

    echo ""
    echo "===== KITTI $seq ====="
    python "$SCRIPT" --image_dir "$IMAGE_DIR" --config "$CONFIG" || {
        echo "${seq},FAIL,FAIL,FAIL" >> "$SUMMARY_CSV"
        echo "KITTI $seq FAILED, continuing..."
        continue
    }

    # Find output dir
    SANITIZED=$(echo "$IMAGE_DIR" | tr '/' '_')
    if [ "$METHOD" = "turboreg" ]; then
        EXP_DIR=$(ls -td /path/to/exps/${SANITIZED}_turboreg/20* 2>/dev/null | head -1)
    else
        EXP_DIR=$(ls -td /path/to/exps/${SANITIZED}/20* 2>/dev/null | head -1)
    fi

    # Eval
    EVAL_DIR="/path/to/eval_results/${MODEL}_${METHOD}_${seq}_${TIMESTAMP}"
    python eval_compare.py \
        --gt_poses "$GT_FILE" --gt_format kitti \
        --${METHOD}_dir "$EXP_DIR" \
        --output_dir "$EVAL_DIR" 2>/dev/null || true

    # Extract to CSV
    EVAL_JSON="${EVAL_DIR}/eval_results.json"
    if [ -f "$EVAL_JSON" ]; then
        python3 -c "
import json
with open('$EVAL_JSON') as f:
    d = json.load(f)
m = d.get('$METHOD', d.get('irls', d.get('turboreg', {})))
def fmt(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) else 'N/A'
print(f'${seq},{fmt(m.get(\"ate_rmse\"))},{fmt(m.get(\"alignment_time\"))},{fmt(m.get(\"total_time\"))}')
" >> "$SUMMARY_CSV"
    else
        echo "${seq},N/A,N/A,N/A" >> "$SUMMARY_CSV"
    fi

    echo "KITTI $seq done"
done

# Print summary
echo ""
echo "============================================================"
echo "COMPLETE: ${MODEL}-Long $METHOD on KITTI"
echo "Summary: $SUMMARY_CSV"
echo "============================================================"
echo ""
printf "%-4s  %12s  %12s  %12s\n" "Seq" "ATE_RMSE" "Align(s)" "Total(s)"
printf "%s\n" "------------------------------------------------"
tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r s a at tt; do
    printf "%-4s  %12s  %12s  %12s\n" "$s" "$a" "$at" "$tt"
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

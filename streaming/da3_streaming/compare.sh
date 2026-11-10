#!/bin/bash
# =============================================================================
# DA3-Streaming: Batch IRLS vs TurboReg on KITTI 00-10, with summary CSV.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash compare.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SUMMARY_CSV="/path/to/eval_results/da3_kitti_summary_${TIMESTAMP}.csv"

mkdir -p /path/to/eval_results

echo "seq,irls_ate_rmse,turboreg_ate_rmse,irls_align_time,turboreg_align_time,irls_total_time,turboreg_total_time" > "$SUMMARY_CSV"

for seq in {00..10}; do
    echo "====================================================="
    echo "Running KITTI sequence: $seq"
    echo "====================================================="

    IMAGE_DIR="/path/to/kitti/sequences/${seq}/image_2"
    POSE_FILE="/path/to/kitti/poses/${seq}.txt"

    if [ ! -d "$IMAGE_DIR" ] || [ ! -f "$POSE_FILE" ]; then
        echo "WARNING: sequence $seq not found, skipping..."
        echo "${seq},N/A,N/A,N/A,N/A,N/A,N/A" >> "$SUMMARY_CSV"
        continue
    fi

    bash "$SCRIPT_DIR/run_compare.sh" "$IMAGE_DIR" "$POSE_FILE" kitti || true

    SANITIZED_DIR=$(echo "$IMAGE_DIR" | tr '/' '_')
    EVAL_JSON=$(ls -t /path/to/eval_results/da3_${SANITIZED_DIR}_*/eval_results.json 2>/dev/null | head -1)

    if [ -z "$EVAL_JSON" ] || [ ! -f "$EVAL_JSON" ]; then
        echo "WARNING: No eval_results.json for seq $seq"
        echo "${seq},N/A,N/A,N/A,N/A,N/A,N/A" >> "$SUMMARY_CSV"
        continue
    fi

    python3 -c "
import json
with open('$EVAL_JSON') as f:
    d = json.load(f)
irls = d.get('irls', {})
tr = d.get('turboreg', {})
def g(obj, key):
    v = obj.get(key)
    return f'{v:.4f}' if v is not None else 'N/A'
print(f'${seq},{g(irls,\"ate_rmse\")},{g(tr,\"ate_rmse\")},{g(irls,\"alignment_time\")},{g(tr,\"alignment_time\")},{g(irls,\"total_time\")},{g(tr,\"total_time\")}')
" >> "$SUMMARY_CSV"

    echo "Sequence $seq done"
    echo ""
done

echo ""
echo "====================================================="
echo "ALL SEQUENCES COMPLETE"
echo "Summary saved to: $SUMMARY_CSV"
echo "====================================================="
echo ""
printf "%-4s  %12s  %12s  %12s  %12s  %12s  %12s\n" \
    "Seq" "IRLS_ATE" "TR_ATE" "IRLS_Align" "TR_Align" "IRLS_Total" "TR_Total"
printf "%s\n" "------------------------------------------------------------------------------------"
tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r seq ia ta iat tat it tt; do
    printf "%-4s  %12s  %12s  %12s  %12s  %12s  %12s\n" \
        "$seq" "$ia" "$ta" "$iat" "$tat" "$it" "$tt"
done
echo ""
python3 -c "
import csv
with open('$SUMMARY_CSV') as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader]
def avg(key):
    vals = [float(r[key]) for r in rows if r[key] != 'N/A']
    return sum(vals)/len(vals) if vals else float('nan')
print(f'AVG   IRLS_ATE={avg(\"irls_ate_rmse\"):.4f}  TR_ATE={avg(\"turboreg_ate_rmse\"):.4f}  '
      f'IRLS_Align={avg(\"irls_align_time\"):.1f}s  TR_Align={avg(\"turboreg_align_time\"):.1f}s  '
      f'IRLS_Total={avg(\"irls_total_time\"):.1f}s  TR_Total={avg(\"turboreg_total_time\"):.1f}s')
"

#!/bin/bash
# =============================================================================
# DA3-Streaming: Batch IRLS vs TurboReg on TUM RGB-D, with summary CSV.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash compare_tum.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUM_ROOT="/path/to/tum"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SUMMARY_CSV="/path/to/eval_results/da3_tum_summary_${TIMESTAMP}.csv"

mkdir -p /path/to/eval_results

SEQUENCES=(
    "rgbd_dataset_freiburg1_360"
    "rgbd_dataset_freiburg1_desk"
    "rgbd_dataset_freiburg1_desk2"
    "rgbd_dataset_freiburg1_floor"
    "rgbd_dataset_freiburg1_plant"
    "rgbd_dataset_freiburg1_room"
    "rgbd_dataset_freiburg1_rpy"
    "rgbd_dataset_freiburg1_teddy"
    "rgbd_dataset_freiburg1_xyz"
)
SHORT_NAMES=("360" "desk" "desk2" "floor" "plant" "room" "rpy" "teddy" "xyz")

echo "seq,irls_ate_rmse,turboreg_ate_rmse,irls_align_time,turboreg_align_time,irls_total_time,turboreg_total_time" > "$SUMMARY_CSV"

for i in "${!SEQUENCES[@]}"; do
    seq="${SEQUENCES[$i]}"
    name="${SHORT_NAMES[$i]}"

    IMAGE_DIR="${TUM_ROOT}/${seq}/rgb"
    GT_FILE="${TUM_ROOT}/${seq}/groundtruth.txt"

    echo "====================================================="
    echo "Running TUM sequence: ${name} (${seq})"
    echo "====================================================="

    if [ ! -d "$IMAGE_DIR" ] || [ ! -f "$GT_FILE" ]; then
        echo "WARNING: ${seq} not found, skipping..."
        echo "${name},N/A,N/A,N/A,N/A,N/A,N/A" >> "$SUMMARY_CSV"
        continue
    fi

    bash "$SCRIPT_DIR/run_compare_tum.sh" "$IMAGE_DIR" "$GT_FILE" || true

    SANITIZED_DIR=$(echo "$IMAGE_DIR" | tr '/' '_')
    EVAL_JSON=$(ls -t /path/to/eval_results/da3_tum_${SANITIZED_DIR}_*/eval_results.json 2>/dev/null | head -1)

    if [ -z "$EVAL_JSON" ] || [ ! -f "$EVAL_JSON" ]; then
        echo "WARNING: No eval_results.json for ${name}"
        echo "${name},N/A,N/A,N/A,N/A,N/A,N/A" >> "$SUMMARY_CSV"
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
print(f'${name},{g(irls,\"ate_rmse\")},{g(tr,\"ate_rmse\")},{g(irls,\"alignment_time\")},{g(tr,\"alignment_time\")},{g(irls,\"total_time\")},{g(tr,\"total_time\")}')
" >> "$SUMMARY_CSV"

    echo "Sequence ${name} done"
    echo ""
done

echo ""
echo "====================================================="
echo "ALL TUM SEQUENCES COMPLETE"
echo "Summary saved to: $SUMMARY_CSV"
echo "====================================================="
echo ""
printf "%-8s  %12s  %12s  %12s  %12s  %12s  %12s\n" \
    "Seq" "IRLS_ATE" "TR_ATE" "IRLS_Align" "TR_Align" "IRLS_Total" "TR_Total"
printf "%s\n" "------------------------------------------------------------------------------------"
tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r seq ia ta iat tat it tt; do
    printf "%-8s  %12s  %12s  %12s  %12s  %12s  %12s\n" \
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

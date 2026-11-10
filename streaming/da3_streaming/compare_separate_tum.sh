#!/bin/bash
# =============================================================================
# DA3-Streaming: Batch TUM RGB-D, run IRLS and TurboReg separately.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash compare_separate_tum.sh
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
    echo "Running TUM: ${name}"
    echo "====================================================="

    if [ ! -d "$IMAGE_DIR" ] || [ ! -f "$GT_FILE" ]; then
        echo "WARNING: ${seq} not found, skipping..."
        echo "${name},N/A,N/A,N/A,N/A,N/A,N/A" >> "$SUMMARY_CSV"
        continue
    fi

    bash "$SCRIPT_DIR/run_separate.sh" "$IMAGE_DIR" "$GT_FILE" tum || true

    SANITIZED_DIR=$(echo "$IMAGE_DIR" | tr '/' '_')
    EVAL_JSON=$(ls -t /path/to/eval_results/da3_${SANITIZED_DIR}_*/eval_results.json 2>/dev/null | head -1)

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
echo "Summary: $SUMMARY_CSV"
echo "====================================================="
echo ""
printf "%-8s  %12s  %12s  %12s  %12s  %12s  %12s\n" \
    "Seq" "IRLS_ATE" "TR_ATE" "IRLS_Align" "TR_Align" "IRLS_Total" "TR_Total"
printf "%s\n" "------------------------------------------------------------------------------------"
tail -n +2 "$SUMMARY_CSV" | while IFS=',' read -r s ia ta iat tat it tt; do
    printf "%-8s  %12s  %12s  %12s  %12s  %12s  %12s\n" \
        "$s" "$ia" "$ta" "$iat" "$tat" "$it" "$tt"
done
echo ""
python3 -c "
import csv
with open('$SUMMARY_CSV') as f:
    rows = [r for r in csv.DictReader(f)]
def avg(key):
    vals = [float(r[key]) for r in rows if r[key] != 'N/A']
    return sum(vals)/len(vals) if vals else float('nan')
ia, ta = avg('irls_ate_rmse'), avg('turboreg_ate_rmse')
ial, tal = avg('irls_align_time'), avg('turboreg_align_time')
it, tt = avg('irls_total_time'), avg('turboreg_total_time')
print(f'AVG   IRLS_ATE={ia:.4f}  TR_ATE={ta:.4f}  IRLS_Align={ial:.1f}s  TR_Align={tal:.1f}s  IRLS_Total={it:.1f}s  TR_Total={tt:.1f}s')
if tal > 0: print(f'Alignment speedup: {ial/tal:.2f}x')
if ta < ia: print(f'ATE improvement: {(1-ta/ia)*100:.1f}% better')
else: print(f'ATE degradation: {(ta/ia-1)*100:.1f}% worse')
"

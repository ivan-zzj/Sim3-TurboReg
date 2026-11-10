#!/bin/bash
# =============================================================================
# Summarize DA3-Streaming alignment time for IRLS vs Sim(3)-TurboReg TSC.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash stat_alignment_time.sh [kitti|tum|all]
#
# Optional:
#   PYTHON_BIN=/path/to/envs/vggt-long/bin/python
#   KITTI_SEQS="00 05 10"       # forwarded to run_batch.sh for KITTI
#   RUN_MODE=collect            # collect latest summaries without running
#
# Output:
#   /path/to/eval_results/da3_alignment_time_<dataset>_<timestamp>.csv
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-/path/to/envs/vggt-long/bin/python}"
RUN_MODE="${RUN_MODE:-run}"
EVAL_ROOT="/path/to/eval_results"
TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"

mkdir -p "$EVAL_ROOT"

if [ "$DATASET" = "all" ]; then
    DATASETS=(kitti tum)
elif [ "$DATASET" = "kitti" ] || [ "$DATASET" = "tum" ]; then
    DATASETS=("$DATASET")
else
    echo "ERROR: dataset must be kitti, tum, or all"
    exit 1
fi

latest_summary() {
    local dataset="$1"
    local method="$2"
    ls -t "${EVAL_ROOT}/da3_${dataset}_${method}_"*.csv 2>/dev/null | head -1
}

run_method() {
    local dataset="$1"
    local method="$2"
    local config_arg=()

    if [ "$method" = "irls" ]; then
        if [ "$dataset" = "tum" ]; then
            config_arg=(configs/tum.yaml)
        else
            config_arg=(configs/base_config.yaml)
        fi
        PYTHON_BIN="$PYTHON_BIN" bash run_batch.sh irls "$dataset" "${config_arg[@]}"
    else
        if [ "$dataset" = "tum" ]; then
            config_arg=(configs/tum_turboreg.yaml)
        else
            config_arg=(configs/turboreg_config.yaml)
        fi
        TURBOREG_USE_DISTANCE_DIFF_CONSISTENCY=false \
        PYTHON_BIN="$PYTHON_BIN" \
        bash run_batch.sh turboreg "$dataset" "${config_arg[@]}"
    fi
}

merge_summaries() {
    local dataset="$1"
    local irls_csv="$2"
    local turboreg_csv="$3"
    local out_csv="${EVAL_ROOT}/da3_alignment_time_${dataset}_${TIMESTAMP}.csv"

    "$PYTHON_BIN" - "$dataset" "$irls_csv" "$turboreg_csv" "$out_csv" <<'PY'
import csv
import math
import sys

dataset, irls_csv, turboreg_csv, out_csv = sys.argv[1:5]

def read_summary(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["seq"]] = row
    return rows

def parse_float(value):
    if value in (None, "", "N/A", "FAIL"):
        return None
    try:
        value = float(value)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value

def fmt(value):
    return "N/A" if value is None else f"{value:.2f}"

irls = read_summary(irls_csv)
turboreg = read_summary(turboreg_csv)
seqs = sorted(set(irls) | set(turboreg))

rows = []
for seq in seqs:
    irls_align = parse_float(irls.get(seq, {}).get("align_time"))
    tr_align = parse_float(turboreg.get(seq, {}).get("align_time"))
    delta = None if irls_align is None or tr_align is None else tr_align - irls_align
    ratio = None if irls_align in (None, 0.0) or tr_align is None else tr_align / irls_align
    rows.append({
        "dataset": dataset,
        "seq": seq,
        "irls_align_time": fmt(irls_align),
        "turboreg_tsc_align_time": fmt(tr_align),
        "delta_turboreg_minus_irls": fmt(delta),
        "ratio_turboreg_over_irls": "N/A" if ratio is None else f"{ratio:.4f}",
    })

def mean(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)

avg_irls = mean(parse_float(irls.get(seq, {}).get("align_time")) for seq in seqs)
avg_tr = mean(parse_float(turboreg.get(seq, {}).get("align_time")) for seq in seqs)
avg_delta = None if avg_irls is None or avg_tr is None else avg_tr - avg_irls
avg_ratio = None if avg_irls in (None, 0.0) or avg_tr is None else avg_tr / avg_irls
rows.append({
    "dataset": dataset,
    "seq": "AVG",
    "irls_align_time": fmt(avg_irls),
    "turboreg_tsc_align_time": fmt(avg_tr),
    "delta_turboreg_minus_irls": fmt(avg_delta),
    "ratio_turboreg_over_irls": "N/A" if avg_ratio is None else f"{avg_ratio:.4f}",
})

fieldnames = [
    "dataset",
    "seq",
    "irls_align_time",
    "turboreg_tsc_align_time",
    "delta_turboreg_minus_irls",
    "ratio_turboreg_over_irls",
]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(out_csv)
print("")
print(f"{dataset.upper()} alignment time summary")
print("-" * 86)
print(f"{'Seq':<10} {'IRLS(s)':>12} {'TurboReg-TSC(s)':>18} {'Delta(s)':>12} {'Ratio':>10}")
print("-" * 86)
for row in rows:
    print(
        f"{row['seq']:<10} "
        f"{row['irls_align_time']:>12} "
        f"{row['turboreg_tsc_align_time']:>18} "
        f"{row['delta_turboreg_minus_irls']:>12} "
        f"{row['ratio_turboreg_over_irls']:>10}"
    )
PY
}

for dataset in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "Dataset: $dataset"
    echo "Mode: $RUN_MODE"
    echo "TurboReg variant: Sim(3)-TurboReg with three-scale consistency"
    echo "============================================================"

    if [ "$RUN_MODE" = "run" ]; then
        echo "[1/2] Running DA3-Streaming IRLS..."
        run_method "$dataset" irls

        echo "[2/2] Running DA3-Streaming Sim(3)-TurboReg TSC..."
        run_method "$dataset" turboreg
    elif [ "$RUN_MODE" = "collect" ]; then
        echo "Collecting latest existing run_batch summaries."
        echo "WARNING: collect mode cannot verify whether the latest TurboReg summary used TSC."
    else
        echo "ERROR: RUN_MODE must be run or collect"
        exit 1
    fi

    IRLS_CSV="$(latest_summary "$dataset" irls || true)"
    TURBOREG_CSV="$(latest_summary "$dataset" turboreg || true)"
    if [ -z "$IRLS_CSV" ] || [ -z "$TURBOREG_CSV" ]; then
        echo "ERROR: Missing summary CSV for $dataset"
        echo "  IRLS: ${IRLS_CSV:-missing}"
        echo "  TurboReg: ${TURBOREG_CSV:-missing}"
        exit 1
    fi

    echo "IRLS summary: $IRLS_CSV"
    echo "TurboReg summary: $TURBOREG_CSV"
    echo "Merged summary:"
    merge_summaries "$dataset" "$IRLS_CSV" "$TURBOREG_CSV"
done

#!/bin/bash
# =============================================================================
# Alignment-only timing for DA3-Streaming IRLS vs Sim(3)-TurboReg TSC.
#
# This script stops immediately after adjacent chunk registration. It does not
# run loop closure, postprocess, point-cloud merge, trajectory eval, or ATE/RPE.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash stat_alignment_time_fast.sh [kitti|tum|all]
#
# Optional:
#   PYTHON_BIN=/path/to/envs/vggt-long/bin/python
#   KITTI_SEQS="00 05 10"
#   TUM_SEQS="360 desk xyz"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

DATASET="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-/path/to/envs/vggt-long/bin/python}"
EVAL_ROOT="/path/to/eval_results"
RUN_ROOT="${EVAL_ROOT}/da3_alignment_only_runs"
TIMESTAMP="$(date +%Y-%m-%d-%H-%M-%S)"

mkdir -p "$EVAL_ROOT" "$RUN_ROOT"

if [ "$DATASET" = "all" ]; then
    DATASETS=(kitti tum)
elif [ "$DATASET" = "kitti" ] || [ "$DATASET" = "tum" ]; then
    DATASETS=("$DATASET")
else
    echo "ERROR: dataset must be kitti, tum, or all"
    exit 1
fi

kitti_sequences() {
    if [ -n "${KITTI_SEQS:-}" ]; then
        read -r -a _seqs <<< "$KITTI_SEQS"
    else
        _seqs=(00 01 02 03 04 05 06 07 08 09 10)
    fi
    printf "%s\n" "${_seqs[@]}"
}

tum_sequences() {
    local all_entries=(
        "360:rgbd_dataset_freiburg1_360"
        "desk:rgbd_dataset_freiburg1_desk"
        "desk2:rgbd_dataset_freiburg1_desk2"
        "floor:rgbd_dataset_freiburg1_floor"
        "plant:rgbd_dataset_freiburg1_plant"
        "room:rgbd_dataset_freiburg1_room"
        "rpy:rgbd_dataset_freiburg1_rpy"
        "teddy:rgbd_dataset_freiburg1_teddy"
        "xyz:rgbd_dataset_freiburg1_xyz"
    )

    if [ -z "${TUM_SEQS:-}" ]; then
        printf "%s\n" "${all_entries[@]}"
        return
    fi

    local selected=()
    read -r -a names <<< "$TUM_SEQS"
    for wanted in "${names[@]}"; do
        for entry in "${all_entries[@]}"; do
            if [ "${entry%%:*}" = "$wanted" ] || [ "${entry##*:}" = "$wanted" ]; then
                selected+=("$entry")
            fi
        done
    done
    printf "%s\n" "${selected[@]}"
}

run_alignment_only() {
    local method="$1"
    local image_dir="$2"
    local config="$3"
    local output_dir="$4"
    local script

    if [ "$method" = "irls" ]; then
        script="da3_streaming.py"
        "$PYTHON_BIN" "$script" \
            --image_dir "$image_dir" \
            --config "$config" \
            --output_dir "$output_dir" \
            --alignment_only
    else
        script="da3_streaming_turboreg.py"
        TURBOREG_USE_DISTANCE_DIFF_CONSISTENCY=false \
        "$PYTHON_BIN" "$script" \
            --image_dir "$image_dir" \
            --config "$config" \
            --output_dir "$output_dir" \
            --alignment_only
    fi
}

read_alignment_time() {
    local timing_json="$1"
    "$PYTHON_BIN" - "$timing_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    timing = json.load(f)
print(f"{float(timing['alignment_time']):.6f}")
PY
}

write_summary() {
    local dataset="$1"
    local csv="$2"

    "$PYTHON_BIN" - "$dataset" "$csv" <<'PY'
import csv
import math
import sys

dataset, csv_path = sys.argv[1:3]
with open(csv_path, newline="") as f:
    rows = list(csv.DictReader(f))

def val(row, key):
    try:
        v = float(row[key])
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v

irls_vals = [val(r, "irls_align_time") for r in rows]
tr_vals = [val(r, "turboreg_tsc_align_time") for r in rows]
irls_vals = [v for v in irls_vals if v is not None]
tr_vals = [v for v in tr_vals if v is not None]
avg_irls = sum(irls_vals) / len(irls_vals) if irls_vals else None
avg_tr = sum(tr_vals) / len(tr_vals) if tr_vals else None
avg_delta = None if avg_irls is None or avg_tr is None else avg_tr - avg_irls
avg_ratio = None if avg_irls in (None, 0.0) or avg_tr is None else avg_tr / avg_irls

print("")
print(f"{dataset.upper()} alignment-only timing")
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
if avg_irls is not None or avg_tr is not None:
    print("-" * 86)
    print(
        f"{'AVG':<10} "
        f"{'N/A' if avg_irls is None else f'{avg_irls:.2f}':>12} "
        f"{'N/A' if avg_tr is None else f'{avg_tr:.2f}':>18} "
        f"{'N/A' if avg_delta is None else f'{avg_delta:.2f}':>12} "
        f"{'N/A' if avg_ratio is None else f'{avg_ratio:.4f}':>10}"
    )
PY
}

run_dataset() {
    local dataset="$1"
    local out_csv="${EVAL_ROOT}/da3_alignment_time_fast_${dataset}_${TIMESTAMP}.csv"
    echo "dataset,seq,irls_align_time,turboreg_tsc_align_time,delta_turboreg_minus_irls,ratio_turboreg_over_irls,irls_output_dir,turboreg_output_dir" > "$out_csv"

    echo "============================================================"
    echo "Dataset: $dataset"
    echo "Mode: alignment-only"
    echo "Python: $PYTHON_BIN"
    echo "TurboReg variant: Sim(3)-TurboReg TSC"
    echo "Summary: $out_csv"
    echo "============================================================"

    if [ "$dataset" = "kitti" ]; then
        while read -r seq; do
            [ -n "$seq" ] || continue
            local image_dir="/path/to/kitti/sequences/${seq}/image_2"
            local gt_file="/path/to/kitti/poses/${seq}.txt"
            if [ ! -d "$image_dir" ] || [ ! -f "$gt_file" ]; then
                echo "WARNING: KITTI $seq not found, skipping..."
                echo "$dataset,$seq,N/A,N/A,N/A,N/A,N/A,N/A" >> "$out_csv"
                continue
            fi
            run_pair "$dataset" "$seq" "$image_dir" "configs/base_config.yaml" "configs/turboreg_config.yaml" "$out_csv"
        done < <(kitti_sequences)
    else
        while read -r entry; do
            [ -n "$entry" ] || continue
            local seq="${entry%%:*}"
            local dirname="${entry##*:}"
            local image_dir="/path/to/tum/${dirname}/rgb"
            local gt_file="/path/to/tum/${dirname}/groundtruth.txt"
            if [ ! -d "$image_dir" ] || [ ! -f "$gt_file" ]; then
                echo "WARNING: TUM $seq not found, skipping..."
                echo "$dataset,$seq,N/A,N/A,N/A,N/A,N/A,N/A" >> "$out_csv"
                continue
            fi
            run_pair "$dataset" "$seq" "$image_dir" "configs/tum.yaml" "configs/tum_turboreg.yaml" "$out_csv"
        done < <(tum_sequences)
    fi

    write_summary "$dataset" "$out_csv"
    echo ""
    echo "Saved: $out_csv"
}

run_pair() {
    local dataset="$1"
    local seq="$2"
    local image_dir="$3"
    local irls_config="$4"
    local turboreg_config="$5"
    local out_csv="$6"

    local irls_out="${RUN_ROOT}/${dataset}_${seq}_irls_${TIMESTAMP}"
    local turboreg_out="${RUN_ROOT}/${dataset}_${seq}_turboreg_tsc_${TIMESTAMP}"

    echo ""
    echo "===== ${dataset^^} $seq: IRLS alignment-only ====="
    run_alignment_only irls "$image_dir" "$irls_config" "$irls_out"

    echo ""
    echo "===== ${dataset^^} $seq: TurboReg TSC alignment-only ====="
    run_alignment_only turboreg "$image_dir" "$turboreg_config" "$turboreg_out"

    local irls_time
    local tr_time
    irls_time="$(read_alignment_time "${irls_out}/timing.json")"
    tr_time="$(read_alignment_time "${turboreg_out}/timing.json")"

    "$PYTHON_BIN" - "$dataset" "$seq" "$irls_time" "$tr_time" "$irls_out" "$turboreg_out" >> "$out_csv" <<'PY'
import sys

dataset, seq, irls_s, tr_s, irls_out, tr_out = sys.argv[1:7]
irls = float(irls_s)
tr = float(tr_s)
delta = tr - irls
ratio = tr / irls if irls else float("nan")
print(f"{dataset},{seq},{irls:.2f},{tr:.2f},{delta:.2f},{ratio:.4f},{irls_out},{tr_out}")
PY
}

for dataset in "${DATASETS[@]}"; do
    run_dataset "$dataset"
done

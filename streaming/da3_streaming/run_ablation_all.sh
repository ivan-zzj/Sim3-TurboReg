#!/bin/bash
# =============================================================================
# Ablation Study: One-click script for the Sim3 streaming ablation table.
#
# Runs ALL experiments sequentially (one sequence at a time to save memory).
# Results are collected into a single CSV for easy copy-paste into LaTeX.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash run_ablation_all.sh
#
# To run a SINGLE experiment group (useful for debugging / resuming):
#   CUDA_VISIBLE_DEVICES=2 bash run_ablation_all.sh --group pipeline
#   CUDA_VISIBLE_DEVICES=2 bash run_ablation_all.sh --group scale_tol
#   CUDA_VISIBLE_DEVICES=2 bash run_ablation_all.sh --group knn
#   CUDA_VISIBLE_DEVICES=2 bash run_ablation_all.sh --group inlier
#   CUDA_VISIBLE_DEVICES=2 bash run_ablation_all.sh --group chunk
#
# To skip certain datasets:
#   CUDA_VISIBLE_DEVICES=2 bash run_ablation_all.sh --skip-tum
#   CUDA_VISIBLE_DEVICES=2 bash run_ablation_all.sh --skip-kitti
# =============================================================================

set -e

export LD_LIBRARY_PATH=/path/to/envs/vggt-long/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

# =============================================================================
# >>>>>>>>>>>>>>>>>>>  USER-CONFIGURABLE PARAMETERS  <<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================

# --- Paths ---
KITTI_SEQ_ROOT="/path/to/kitti/sequences"
KITTI_POSE_ROOT="/path/to/kitti/poses"
TUM_ROOT="/path/to/tum"
RESULT_ROOT="/path/to/eval_results"
EXP_ROOT="/path/to/exps"

DA3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VGGT_DIR="${VGGT_DIR:-$(cd "$DA3_DIR/../vggt_long" && pwd)}"

# --- Default TurboReg parameters (used as baseline; ablation changes one at a time) ---
DEFAULT_CHUNK_SIZE=60
DEFAULT_OVERLAP=30          # overlap = chunk_size / 2
DEFAULT_RADIU_NMS=0.15      # Scale Tolerance (tau)
DEFAULT_K_NEIGHBORS=20      # KNN k
DEFAULT_TAU_INLIER=0.01     # Inlier threshold (epsilon)

# --- KITTI sequences ---
KITTI_SEQS=(00 01 02 03 04 05 06 07 08 09 10)

# --- TUM sequences (folder:short_name) ---
TUM_SEQS=(
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

# =============================================================================
# >>>>>>>>>>>>>>>>>>>  END USER-CONFIGURABLE PARAMETERS  <<<<<<<<<<<<<<<<<<<<<
# =============================================================================

# --- Parse CLI args ---
RUN_GROUP="all"
SKIP_KITTI=false
SKIP_TUM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)    RUN_GROUP="$2"; shift 2 ;;
        --skip-kitti) SKIP_KITTI=true; shift ;;
        --skip-tum)   SKIP_TUM=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_CSV="${RESULT_ROOT}/ablation_master_${TIMESTAMP}.csv"
mkdir -p "$RESULT_ROOT"

echo "experiment,group,variant,dataset,seq,ate_rmse,total_time_s,num_images" > "$MASTER_CSV"

echo "============================================================"
echo "  Ablation Study - Master Script"
echo "  Timestamp: $TIMESTAMP"
echo "  Group:     $RUN_GROUP"
echo "  Results:   $MASTER_CSV"
echo "============================================================"

# =============================================================================
# Helper: Generate a temporary TurboReg config for DA3-Streaming ablation
# =============================================================================
generate_da3_turboreg_config() {
    local out_path="$1"
    local chunk_size="$2"
    local overlap="$3"
    local radiu_nms="$4"
    local k_neighbors="$5"
    local tau_inlier="$6"

    cat > "$out_path" <<YAMLEOF
Weights:
  DA3: '/path/to/weights/da3/model.safetensors'
  DA3_CONFIG: '/path/to/weights/da3/config.json'
  SALAD: '/path/to/weights/dino_salad.ckpt'

Model:
  chunk_size: ${chunk_size}
  overlap: ${overlap}
  loop_chunk_size: 20
  loop_enable: True
  useDBoW: False
  delete_temp_files: True
  align_lib: 'turboreg'
  align_method: 'sim3'
  scale_compute_method: 'auto'
  align_type: 'dense'

  ref_view_strategy: 'saddle_balanced'
  ref_view_strategy_loop: 'saddle_balanced'
  depth_threshold: 15.0

  save_depth_conf_result: True
  save_debug_info: False

  Sparse_Align:
    keypoint_select: 'orb'
    keypoint_num: 5000

  IRLS:
    delta: 0.1
    max_iters: 5
    tol: 1e-9

  Pointcloud_Save:
    sample_ratio: 0.015
    conf_threshold_coef: 0.75

TurboReg:
  max_N: 7000
  epsilon_scale: 0.25
  num_pivot: 2000
  radiu_nms: ${radiu_nms}
  tau_inlier: ${tau_inlier}
  metric_str: 'IN'
  k_neighbors: ${k_neighbors}
  num_iterations: 2
  graph_k: 80
  trim_ratio: 0.1
  min_pair_dist: 0.004
  local_scale_min: 0.5
  local_scale_max: 2.0
  local_scale_diff: -1.0
  local_scale_mode: 'union_median'

Loop:
  SALAD:
    image_size: [336, 336]
    batch_size: 32
    similarity_threshold: 0.85
    top_k: 5
    use_nms: True
    nms_threshold: 25

  SIM3_Optimizer:
    lang_version: 'cpp'
    max_iterations: 30
    lambda_init: 1e-6
YAMLEOF
}

# Same but for TUM (uses different ref_view_strategy for TUM if needed — currently same)
generate_da3_turboreg_config_tum() {
    # For now, TUM uses the same config structure. Adjust if needed.
    generate_da3_turboreg_config "$@"
}

# =============================================================================
# Helper: Run ONE sequence and collect results
# =============================================================================
run_one_sequence() {
    local experiment_name="$1"  # e.g., "pipeline_da3-streaming"
    local group="$2"            # e.g., "pipeline"
    local variant="$3"          # e.g., "da3-streaming"
    local dataset="$4"          # "kitti" or "tum"
    local seq_name="$5"         # e.g., "00" or "desk"
    local image_dir="$6"
    local gt_file="$7"
    local gt_format="$8"        # "kitti" or "tum"
    local script_dir="$9"       # directory containing the python script
    local script="${10}"         # python script name
    local config="${11}"         # config file path

    if [ ! -d "$image_dir" ]; then
        echo "  [SKIP] $image_dir not found"
        echo "${experiment_name},${group},${variant},${dataset},${seq_name},N/A,N/A,N/A" >> "$MASTER_CSV"
        return
    fi
    if [ ! -f "$gt_file" ]; then
        echo "  [SKIP] $gt_file not found"
        echo "${experiment_name},${group},${variant},${dataset},${seq_name},N/A,N/A,N/A" >> "$MASTER_CSV"
        return
    fi

    echo "  >>> ${dataset}/${seq_name} ..."

    # Run inference
    local run_ok=true
    (cd "$script_dir" && python "$script" --image_dir "$image_dir" --config "$config") || run_ok=false

    if [ "$run_ok" = false ]; then
        echo "  [FAIL] ${dataset}/${seq_name}"
        echo "${experiment_name},${group},${variant},${dataset},${seq_name},FAIL,FAIL,FAIL" >> "$MASTER_CSV"
        return
    fi

    # Find the latest output directory
    local sanitized
    sanitized=$(echo "$image_dir" | tr '/' '_')
    local exp_dir
    exp_dir=$(ls -td "${EXP_ROOT}/${sanitized}_turboreg"/20* "${EXP_ROOT}/${sanitized}"/20* 2>/dev/null | head -1)

    if [ -z "$exp_dir" ]; then
        echo "  [WARN] No output dir found for ${dataset}/${seq_name}"
        echo "${experiment_name},${group},${variant},${dataset},${seq_name},NO_OUTPUT,NO_OUTPUT,NO_OUTPUT" >> "$MASTER_CSV"
        return
    fi

    # Run evaluation
    local eval_dir="${RESULT_ROOT}/${experiment_name}_${dataset}_${seq_name}_${TIMESTAMP}"
    (cd "$DA3_DIR" && python eval_compare.py \
        --gt_poses "$gt_file" --gt_format "$gt_format" \
        --irls_dir "$exp_dir" \
        --output_dir "$eval_dir" 2>&1) || true

    # Also try turboreg_dir if irls_dir fails
    local eval_json="${eval_dir}/eval_results.json"
    if [ ! -f "$eval_json" ]; then
        (cd "$DA3_DIR" && python eval_compare.py \
            --gt_poses "$gt_file" --gt_format "$gt_format" \
            --turboreg_dir "$exp_dir" \
            --output_dir "$eval_dir" 2>&1) || true
    fi

    # Extract results
    if [ -f "$eval_json" ]; then
        python3 -c "
import json
with open('${eval_json}') as f:
    d = json.load(f)
# Try to find the method result (could be under any key)
m = None
for key in ['irls', 'turboreg', 'da3', 'vggt', 'pi3']:
    if key in d and d[key] is not None:
        m = d[key]
        break
if m is None:
    m = list(d.values())[0] if d else {}
ate = m.get('ate_rmse', 'N/A')
tt = m.get('total_time', 'N/A')
ni = m.get('num_poses', 'N/A')
ate_s = f'{ate:.4f}' if isinstance(ate, (int, float)) else ate
tt_s = f'{tt:.2f}' if isinstance(tt, (int, float)) else tt
print(f'${experiment_name},${group},${variant},${dataset},${seq_name},{ate_s},{tt_s},{ni}')
" >> "$MASTER_CSV"
    else
        echo "${experiment_name},${group},${variant},${dataset},${seq_name},N/A,N/A,N/A" >> "$MASTER_CSV"
    fi
}

# =============================================================================
# Helper: Run all KITTI sequences for an experiment
# =============================================================================
run_kitti() {
    local experiment_name="$1"
    local group="$2"
    local variant="$3"
    local script_dir="$4"
    local script="$5"
    local config="$6"

    if [ "$SKIP_KITTI" = true ]; then
        echo "  [SKIP-KITTI] $experiment_name"
        return
    fi

    for seq in "${KITTI_SEQS[@]}"; do
        run_one_sequence \
            "$experiment_name" "$group" "$variant" \
            "kitti" "$seq" \
            "${KITTI_SEQ_ROOT}/${seq}/image_2" \
            "${KITTI_POSE_ROOT}/${seq}.txt" \
            "kitti" \
            "$script_dir" "$script" "$config"
    done
}

# =============================================================================
# Helper: Run all TUM sequences for an experiment
# =============================================================================
run_tum() {
    local experiment_name="$1"
    local group="$2"
    local variant="$3"
    local script_dir="$4"
    local script="$5"
    local config="$6"

    if [ "$SKIP_TUM" = true ]; then
        echo "  [SKIP-TUM] $experiment_name"
        return
    fi

    for entry in "${TUM_SEQS[@]}"; do
        local folder="${entry%%:*}"
        local name="${entry##*:}"
        run_one_sequence \
            "$experiment_name" "$group" "$variant" \
            "tum" "$name" \
            "${TUM_ROOT}/${folder}/rgb" \
            "${TUM_ROOT}/${folder}/groundtruth.txt" \
            "tum" \
            "$script_dir" "$script" "$config"
    done
}

# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

# ---- Group 1: Pipeline ----
run_group_pipeline() {
    echo ""
    echo "============================================================"
    echo "  Group 1/5: Pipeline (vggt-long, pi3-long, da3-streaming)"
    echo "============================================================"

    # --- vggt-long (IRLS) ---
    echo ""
    echo "--- [1/3] vggt-long ---"
    run_kitti "pipeline_vggt-long" "pipeline" "vggt-long" \
        "$VGGT_DIR" "vggt_long.py" "${VGGT_DIR}/configs/base_config.yaml"
    run_tum   "pipeline_vggt-long" "pipeline" "vggt-long" \
        "$VGGT_DIR" "vggt_long.py" "${VGGT_DIR}/configs/base_config.yaml"

    # --- pi3-long (IRLS) ---
    echo ""
    echo "--- [2/3] pi3-long ---"
    run_kitti "pipeline_pi3-long" "pipeline" "pi3-long" \
        "$VGGT_DIR" "vggt_long.py" "${VGGT_DIR}/configs/pi3_base_config.yaml"
    run_tum   "pipeline_pi3-long" "pipeline" "pi3-long" \
        "$VGGT_DIR" "vggt_long.py" "${VGGT_DIR}/configs/pi3_base_config.yaml"

    # --- da3-streaming (TurboReg, default config) ---
    echo ""
    echo "--- [3/3] da3-streaming (default) ---"
    local cfg="/tmp/ablation_da3_default_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg" \
        "$DEFAULT_CHUNK_SIZE" "$DEFAULT_OVERLAP" \
        "$DEFAULT_RADIU_NMS" "$DEFAULT_K_NEIGHBORS" "$DEFAULT_TAU_INLIER"
    run_kitti "pipeline_da3-streaming" "pipeline" "da3-streaming" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"
    run_tum   "pipeline_da3-streaming" "pipeline" "da3-streaming" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"
}

# ---- Group 2: Scale Tolerance (radiu_nms / tau) ----
run_group_scale_tol() {
    echo ""
    echo "============================================================"
    echo "  Group 2/5: Scale Tolerance (tau)"
    echo "============================================================"

    # tau = 0.2
    echo ""
    echo "--- Scale Tol = 0.2 ---"
    local cfg="/tmp/ablation_scale_0.2_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg" \
        "$DEFAULT_CHUNK_SIZE" "$DEFAULT_OVERLAP" \
        0.2 "$DEFAULT_K_NEIGHBORS" "$DEFAULT_TAU_INLIER"
    run_kitti "scale_tol_0.2" "scale_tol" "0.2" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"
    run_tum   "scale_tol_0.2" "scale_tol" "0.2" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"

    # tau = 0.15 (default — same as pipeline default, skip if already run)
    echo ""
    echo "--- Scale Tol = 0.15 (default) ---"
    local cfg2="/tmp/ablation_scale_0.15_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg2" \
        "$DEFAULT_CHUNK_SIZE" "$DEFAULT_OVERLAP" \
        0.15 "$DEFAULT_K_NEIGHBORS" "$DEFAULT_TAU_INLIER"
    run_kitti "scale_tol_0.15" "scale_tol" "0.15" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg2"
    run_tum   "scale_tol_0.15" "scale_tol" "0.15" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg2"
}

# ---- Group 3: KNN (k_neighbors) ----
run_group_knn() {
    echo ""
    echo "============================================================"
    echo "  Group 3/5: KNN (k)"
    echo "============================================================"

    # k = 20 (default — same as pipeline default)
    echo ""
    echo "--- KNN k=20 (default) ---"
    local cfg="/tmp/ablation_knn_20_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg" \
        "$DEFAULT_CHUNK_SIZE" "$DEFAULT_OVERLAP" \
        "$DEFAULT_RADIU_NMS" 20 "$DEFAULT_TAU_INLIER"
    run_kitti "knn_20" "knn" "20" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"
    run_tum   "knn_20" "knn" "20" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"

    # k = 25
    echo ""
    echo "--- KNN k=25 ---"
    local cfg2="/tmp/ablation_knn_25_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg2" \
        "$DEFAULT_CHUNK_SIZE" "$DEFAULT_OVERLAP" \
        "$DEFAULT_RADIU_NMS" 25 "$DEFAULT_TAU_INLIER"
    run_kitti "knn_25" "knn" "25" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg2"
    run_tum   "knn_25" "knn" "25" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg2"
}

# ---- Group 4: Inlier Threshold (tau_inlier / epsilon) ----
run_group_inlier() {
    echo ""
    echo "============================================================"
    echo "  Group 4/5: Inlier Threshold (epsilon)"
    echo "============================================================"

    # epsilon = 0.01 (default)
    echo ""
    echo "--- Inlier = 0.01 (default) ---"
    local cfg="/tmp/ablation_inlier_0.01_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg" \
        "$DEFAULT_CHUNK_SIZE" "$DEFAULT_OVERLAP" \
        "$DEFAULT_RADIU_NMS" "$DEFAULT_K_NEIGHBORS" 0.01
    run_kitti "inlier_0.01" "inlier" "0.01" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"
    run_tum   "inlier_0.01" "inlier" "0.01" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"

    # epsilon = 0.02
    echo ""
    echo "--- Inlier = 0.02 ---"
    local cfg2="/tmp/ablation_inlier_0.02_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg2" \
        "$DEFAULT_CHUNK_SIZE" "$DEFAULT_OVERLAP" \
        "$DEFAULT_RADIU_NMS" "$DEFAULT_K_NEIGHBORS" 0.02
    run_kitti "inlier_0.02" "inlier" "0.02" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg2"
    run_tum   "inlier_0.02" "inlier" "0.02" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg2"
}

# ---- Group 5: Chunk Size ----
run_group_chunk() {
    echo ""
    echo "============================================================"
    echo "  Group 5/5: Chunk Size"
    echo "============================================================"

    # chunk_size = 60 (default)
    echo ""
    echo "--- Chunk Size = 60 (default) ---"
    local cfg="/tmp/ablation_chunk_60_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg" \
        60 30 \
        "$DEFAULT_RADIU_NMS" "$DEFAULT_K_NEIGHBORS" "$DEFAULT_TAU_INLIER"
    run_kitti "chunk_60" "chunk" "60" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"
    run_tum   "chunk_60" "chunk" "60" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg"

    # chunk_size = 120
    echo ""
    echo "--- Chunk Size = 120 ---"
    local cfg2="/tmp/ablation_chunk_120_${TIMESTAMP}.yaml"
    generate_da3_turboreg_config "$cfg2" \
        120 60 \
        "$DEFAULT_RADIU_NMS" "$DEFAULT_K_NEIGHBORS" "$DEFAULT_TAU_INLIER"
    run_kitti "chunk_120" "chunk" "120" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg2"
    run_tum   "chunk_120" "chunk" "120" \
        "$DA3_DIR" "da3_streaming_turboreg.py" "$cfg2"
}

# =============================================================================
# MAIN: Run selected groups
# =============================================================================

case "$RUN_GROUP" in
    all)
        run_group_pipeline
        run_group_scale_tol
        run_group_knn
        run_group_inlier
        run_group_chunk
        ;;
    pipeline)   run_group_pipeline ;;
    scale_tol)  run_group_scale_tol ;;
    knn)        run_group_knn ;;
    inlier)     run_group_inlier ;;
    chunk)      run_group_chunk ;;
    *)
        echo "Unknown group: $RUN_GROUP"
        echo "Valid groups: all, pipeline, scale_tol, knn, inlier, chunk"
        exit 1
        ;;
esac

# =============================================================================
# SUMMARY: Compute averages and print final table
# =============================================================================
echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Raw CSV: $MASTER_CSV"
echo "============================================================"
echo ""

python3 << 'PYEOF'
import csv
import sys
from collections import defaultdict

csv_path = sys.argv[1] if len(sys.argv) > 1 else ""
if not csv_path:
    # Read from env
    import os
    csv_path = os.environ.get("MASTER_CSV", "")

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

# Group by (group, variant, dataset)
grouped = defaultdict(list)
for r in rows:
    key = (r['group'], r['variant'], r['dataset'])
    grouped[key].append(r)

def safe_float(v):
    try:
        return float(v)
    except:
        return None

# Print the LaTeX-style summary table
print(f"{'Group':<15} {'Variant':<20} {'Dataset':<8} {'Avg ATE':>10} {'Avg* ATE':>10} {'Avg T(ms)':>10} {'Avg* T(ms)':>10}")
print("-" * 95)

seen_groups = []
for (group, variant, dataset), recs in sorted(grouped.items()):
    ates = []
    ates_star = []  # excluding KITTI seq 01
    times_ms = []
    times_ms_star = []

    for r in recs:
        ate = safe_float(r['ate_rmse'])
        tt = safe_float(r['total_time_s'])
        ni = safe_float(r['num_images'])

        if ate is not None:
            ates.append(ate)
            if not (dataset == 'kitti' and r['seq'] == '01'):
                ates_star.append(ate)

        if tt is not None and ni is not None and ni > 0:
            t_ms = tt / ni * 1000
            times_ms.append(t_ms)
            if not (dataset == 'kitti' and r['seq'] == '01'):
                times_ms_star.append(t_ms)

    avg_ate = f"{sum(ates)/len(ates):.2f}" if ates else "N/A"
    avg_ate_star = f"{sum(ates_star)/len(ates_star):.2f}" if ates_star else "N/A"
    avg_t = f"{sum(times_ms)/len(times_ms):.1f}" if times_ms else "N/A"
    avg_t_star = f"{sum(times_ms_star)/len(times_ms_star):.1f}" if times_ms_star else "N/A"

    # For TUM, Avg* is same as Avg
    if dataset == 'tum':
        avg_ate_star = "-"
        avg_t_star = "-"

    print(f"{group:<15} {variant:<20} {dataset:<8} {avg_ate:>10} {avg_ate_star:>10} {avg_t:>10} {avg_t_star:>10}")

print()
print(f"Full per-sequence results: {csv_path}")
PYEOF
"$MASTER_CSV"

echo ""
echo "Done! Results saved to: $MASTER_CSV"

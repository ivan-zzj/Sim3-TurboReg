"""
Minimal evaluation: IRLS vs TurboReg alignment quality.

Only runs VGGT inference to produce chunk data, then compares alignment methods.
No loop closure, no point cloud saving, no full pipeline.

Outputs per-pair and aggregate metrics:
  - Alignment residual (mean/median/max)
  - Scale estimate
  - Rotation angle difference
  - Translation difference
  - Time per pair

Usage:
    # Use pre-computed chunks:
    python eval_alignment_only.py \
        --chunks_dir /path/to/_tmp_results_unaligned \
        --overlap 30 \
        --output_dir /path/to/data/eval_align

    # Generate chunks from images (inference only, no full pipeline):
    python eval_alignment_only.py \
        --image_dir /path/to/kitti/sequences/04/image_2 \
        --config configs/turboreg_config.yaml \
        --output_dir /path/to/data/eval_align

    # With GT poses for absolute comparison:
    python eval_alignment_only.py \
        --image_dir /path/to/kitti/sequences/04/image_2 \
        --config configs/turboreg_config.yaml \
        --gt_poses /path/to/kitti/poses/04.txt \
        --output_dir /path/to/data/eval_align
"""

import argparse
import json
import os
import sys
import time
import glob

import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = os.path.dirname(os.path.abspath(__file__))
base_models_path = os.path.join(current_dir, 'base_models')
if base_models_path not in sys.path:
    sys.path.append(base_models_path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Inference: generate chunks (minimal, no loop / no pointcloud)
# ---------------------------------------------------------------------------

def generate_chunks(image_dir, config, output_dir):
    """Run VGGT inference only, save chunk .npy files."""
    import torch
    from base_models.base_model import VGGTAdapter, Pi3Adapter, MapAnythingAdapter

    chunk_size = config['Model']['chunk_size']
    overlap = config['Model']['overlap']

    img_list = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                      glob.glob(os.path.join(image_dir, "*.png")))
    if not img_list:
        raise ValueError(f"No images in {image_dir}")
    print(f"Found {len(img_list)} images")

    # Compute chunk indices
    if len(img_list) <= chunk_size:
        chunk_indices = [(0, len(img_list))]
    else:
        step = chunk_size - overlap
        chunk_indices = []
        i = 0
        while i + chunk_size <= len(img_list):
            chunk_indices.append((i, i + chunk_size))
            i += step
        if chunk_indices[-1][1] < len(img_list):
            chunk_indices.append((len(img_list) - chunk_size, len(img_list)))

    print(f"{len(chunk_indices)} chunks (size={chunk_size}, overlap={overlap})")

    # Load model
    model_name = config['Weights']['model']
    if model_name == 'VGGT':
        model = VGGTAdapter(config)
    elif model_name == 'Pi3':
        model = Pi3Adapter(config)
    elif model_name == 'Mapanything':
        model = MapAnythingAdapter(config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.load()

    os.makedirs(output_dir, exist_ok=True)

    for idx, (s, e) in enumerate(chunk_indices):
        save_path = os.path.join(output_dir, f"chunk_{idx}.npy")
        if os.path.exists(save_path):
            print(f"  chunk_{idx} already exists, skip")
            continue
        print(f"  Inferring chunk {idx}/{len(chunk_indices)-1} [{s}:{e}]")
        predictions = model.infer_chunk(img_list[s:e])
        for key in predictions:
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        predictions['depth'] = np.squeeze(predictions['depth'])
        np.save(save_path, predictions)
        torch.cuda.empty_cache()

    del model
    import torch
    torch.cuda.empty_cache()

    return chunk_indices, img_list


# ---------------------------------------------------------------------------
# Alignment methods
# ---------------------------------------------------------------------------

def extract_correspondences(pm1, c1, pm2, c2, mask, conf_threshold):
    """Extract filtered point correspondences from overlapping regions."""
    b = min(pm1.shape[0], pm2.shape[0])
    pts1_list, pts2_list, w_list = [], [], []

    for i in range(b):
        m1 = c1[i] > conf_threshold
        m2 = c2[i] > conf_threshold
        valid = m1 & m2
        if mask is not None:
            valid = valid & mask[i].squeeze()
        idx = np.where(valid)
        if len(idx[0]) == 0:
            continue
        p1, p2 = pm1[i][idx], pm2[i][idx]
        ok = np.isfinite(p1).all(1) & np.isfinite(p2).all(1)
        pts1_list.append(p1[ok])
        pts2_list.append(p2[ok])
        w_list.append(np.sqrt(c1[i][idx][ok] * c2[i][idx][ok]))

    if not pts1_list:
        raise ValueError("No correspondences found")
    return (np.concatenate(pts1_list),
            np.concatenate(pts2_list),
            np.concatenate(w_list))


def run_irls(pts1, pts2, weights, config):
    """IRLS alignment (original method)."""
    from loop_utils.sim3utils import (
        robust_weighted_estimate_sim3_numba,
        robust_weighted_estimate_sim3,
    )
    use_sim3 = config['Model']['using_sim3']
    delta = config['Model']['IRLS']['delta']
    max_iters = config['Model']['IRLS']['max_iters']
    tol = eval(config['Model']['IRLS']['tol'])

    if config['Model']['align_method'] == 'numba':
        s, R, t = robust_weighted_estimate_sim3_numba(
            pts2, pts1, weights, delta=delta, max_iters=max_iters,
            tol=tol, using_sim3=use_sim3)
    else:
        s, R, t = robust_weighted_estimate_sim3(
            pts2, pts1, weights, delta=delta, max_iters=max_iters,
            tol=tol, using_sim3=use_sim3)
    return s, R, t


def run_turboreg(pts1, pts2, reger, device, max_n=7000):
    """TurboReg+Umeyama alignment (proposed method)."""
    import torch

    src, tgt = pts2.copy(), pts1.copy()  # src=pts2 -> tgt=pts1
    if len(src) > max_n:
        idx = np.random.permutation(len(src))[:max_n]
        src, tgt = src[idx], tgt[idx]

    kpts_src = torch.from_numpy(src.astype(np.float32)).to(device)
    kpts_tgt = torch.from_numpy(tgt.astype(np.float32)).to(device)
    trans = reger.run_reg(kpts_src, kpts_tgt).cpu().numpy()

    sR = trans[:3, :3]
    t = trans[:3, 3]
    s = np.linalg.norm(sR[:, 0])
    if s > 1e-6:
        R = sR / s
    else:
        s, R = 1.0, np.eye(3)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return s, R, t


def run_teaser(pts1, pts2, config, max_n=7000):
    """TEASER++ alignment."""
    import teaserpp_python

    src, tgt = pts2.copy(), pts1.copy()  # src=pts2 -> tgt=pts1
    if len(src) > max_n:
        idx = np.random.permutation(len(src))[:max_n]
        src, tgt = src[idx], tgt[idx]

    teaser_cfg = config.get('TEASER', {})
    noise_bound = teaser_cfg.get('noise_bound', 0.05)

    solver_params = teaserpp_python.RobustRegistrationSolverParams(
        cbar2=teaser_cfg.get('cbar2', 1.0),
        noise_bound=noise_bound,
        estimate_scaling=True,
        rotation_estimation_algorithm=teaserpp_python.RotationEstimationAlgorithm.GNC_TLS,
        rotation_gnc_factor=teaser_cfg.get('rotation_gnc_factor', 1.4),
        rotation_max_iterations=teaser_cfg.get('rotation_max_iterations', 100),
        rotation_cost_threshold=teaser_cfg.get('rotation_cost_threshold', 1e-6),
    )
    solver = teaserpp_python.RobustRegistrationSolver(*solver_params)

    # TEASER++ expects 3xN arrays
    solver.solve(src.T.astype(np.float64), tgt.T.astype(np.float64))
    solution = solver.solution

    s = float(solution.scale)
    R = solution.rotation
    t = solution.translation
    return s, R, t


def compute_residual(pts1, pts2, s, R, t, conf_threshold=None):
    """Mean residual after applying Sim(3): ||pts1 - (s*R@pts2 + t)||"""
    transformed = s * (pts2 @ R.T) + t
    residuals = np.linalg.norm(pts1 - transformed, axis=1)
    return {
        'mean': float(np.mean(residuals)),
        'median': float(np.median(residuals)),
        'max': float(np.max(residuals)),
        'std': float(np.std(residuals)),
        'p90': float(np.percentile(residuals, 90)),
    }


def rotation_angle(R):
    """Rotation angle in degrees."""
    cos_a = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_a)))


def rotation_diff(R1, R2):
    """Angle between two rotations in degrees."""
    return rotation_angle(R1 @ R2.T)


# ---------------------------------------------------------------------------
# GT evaluation helpers
# ---------------------------------------------------------------------------

def load_kitti_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return poses


def umeyama_align(src, dst):
    """Umeyama Sim(3) alignment. Returns s, R, t."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = float(np.mean(np.sum(X ** 2, axis=1)))
    scale = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def compute_ate(pred_positions, gt_positions):
    """ATE after Umeyama alignment."""
    s, R, t = umeyama_align(pred_positions, gt_positions)
    aligned = s * (pred_positions @ R.T) + t
    errors = np.linalg.norm(aligned - gt_positions, axis=1)
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'max': float(np.max(errors)),
    }, aligned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare IRLS vs TurboReg alignment on VGGT chunks')
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--chunks_dir', type=str, default=None,
                     help='Pre-computed chunks directory')
    grp.add_argument('--image_dir', type=str, default=None,
                     help='Image directory (will run inference)')
    parser.add_argument('--config', type=str, default='./configs/turboreg_config.yaml')
    parser.add_argument('--overlap', type=int, default=None,
                        help='Override overlap (default: from config)')
    parser.add_argument('--gt_poses', type=str, default=None,
                        help='KITTI GT poses for ATE eval')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/data/eval_align')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    from loop_utils.config_utils import load_config
    config = load_config(args.config)

    overlap = args.overlap or config['Model']['overlap']
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Get chunks ----
    if args.chunks_dir:
        chunks_dir = args.chunks_dir
    else:
        chunks_dir = os.path.join(args.output_dir, '_chunks')
        print("=" * 60)
        print("Step 1: VGGT Inference (generating chunks)")
        print("=" * 60)
        generate_chunks(args.image_dir, config, chunks_dir)
        print()

    chunk_files = sorted([f for f in os.listdir(chunks_dir)
                          if f.startswith('chunk_') and f.endswith('.npy')])
    num_chunks = len(chunk_files)
    print(f"Found {num_chunks} chunks in {chunks_dir}")
    if num_chunks < 2:
        print("Need >= 2 chunks")
        return 1

    # ---- Warmup IRLS numba ----
    if config['Model']['align_method'] == 'numba':
        from loop_utils.sim3utils import warmup_numba
        warmup_numba()

    # ---- Init TurboReg ----
    import torch
    import turboreg_gpu

    tr_cfg = config.get('TurboReg', {})
    reger = turboreg_gpu.ScaleTurboRegGPU(
        tr_cfg.get('max_N', 7000),
        tr_cfg.get('epsilon_scale', 0.1),
        tr_cfg.get('num_pivot', 2000),
        tr_cfg.get('radiu_nms', 0.15),
        tr_cfg.get('tau_inlier', 0.1),
        tr_cfg.get('metric_str', 'IN'),
        tr_cfg.get('k_neighbors', 20),
        tr_cfg.get('num_iterations', 2),
        tr_cfg.get('graph_k', 20),
        tr_cfg.get('trim_ratio', 0.1),
        tr_cfg.get('min_pair_dist', 1e-3),
        tr_cfg.get('local_scale_min', 0.1),
        tr_cfg.get('local_scale_max', 10.0),
        tr_cfg.get('local_scale_diff', -1.0),
        tr_cfg.get('local_scale_mode', 'hist'),
    )
    tr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Run alignment comparison ----
    print()
    print("=" * 60)
    print("Step 2: Alignment Comparison (IRLS vs TurboReg)")
    print("=" * 60)

    # Method registry: name -> (results_list, sim3_list)
    methods = {
        'IRLS': {'all': [], 'sim3': []},
        'TurboReg': {'all': [], 'sim3': []},
        # 'TEASER++': {'all': [], 'sim3': []},  # temporarily disabled
    }

    # teaser_max_n = config.get('TEASER', {}).get('max_N', 7000)  # temporarily disabled

    for ci in range(num_chunks - 1):
        print(f"\n--- Pair {ci} <-> {ci+1} ---")
        d1 = np.load(os.path.join(chunks_dir, f"chunk_{ci}.npy"), allow_pickle=True).item()
        d2 = np.load(os.path.join(chunks_dir, f"chunk_{ci+1}.npy"), allow_pickle=True).item()

        pm1 = d1['world_points'][-overlap:]
        pm2 = d2['world_points'][:overlap]
        c1 = d1['world_points_conf'][-overlap:]
        c2 = d2['world_points_conf'][:overlap]

        mask = None
        if d1.get("mask") is not None and d1["mask"] is not None:
            mask = d1["mask"][-overlap:].squeeze() & d2["mask"][:overlap].squeeze()

        # Free full chunk data, keep only overlap slices
        del d1, d2

        # Delete chunk file that is no longer needed (chunk ci is fully consumed)
        if not args.chunks_dir:
            old_chunk = os.path.join(chunks_dir, f"chunk_{ci}.npy")
            if os.path.exists(old_chunk):
                os.remove(old_chunk)
                print(f"  Deleted {old_chunk}")

        if config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
            conf_threshold = min(np.median(c1), np.median(c2)) * 0.1
        else:
            conf_threshold = -1.0

        pts1, pts2, weights = extract_correspondences(pm1, c1, pm2, c2, mask, conf_threshold)
        del pm1, pm2, c1, c2, mask
        print(f"  Correspondences: {len(pts1)}")

        # ---- IRLS ----
        t0 = time.time()
        s_i, R_i, t_i = run_irls(pts1, pts2, weights, config)
        irls_time = time.time() - t0
        irls_res = compute_residual(pts1, pts2, s_i, R_i, t_i)
        methods['IRLS']['all'].append({
            'pair': f"{ci}-{ci+1}", 's': float(s_i), 'time': irls_time,
            'rot_angle': rotation_angle(R_i), **irls_res,
        })
        methods['IRLS']['sim3'].append((s_i, R_i, t_i))

        # ---- TurboReg ----
        t0 = time.time()
        s_t, R_t, t_t = run_turboreg(pts1, pts2, reger, tr_device,
                                       max_n=tr_cfg.get('max_N', 7000))
        tr_time = time.time() - t0
        tr_res = compute_residual(pts1, pts2, s_t, R_t, t_t)
        methods['TurboReg']['all'].append({
            'pair': f"{ci}-{ci+1}", 's': float(s_t), 'time': tr_time,
            'rot_angle': rotation_angle(R_t), **tr_res,
        })
        methods['TurboReg']['sim3'].append((s_t, R_t, t_t))

        # ---- TEASER++ ---- (temporarily disabled)
        # t0 = time.time()
        # s_te, R_te, t_te = run_teaser(pts1, pts2, config, max_n=teaser_max_n)
        # teaser_time = time.time() - t0
        # teaser_res = compute_residual(pts1, pts2, s_te, R_te, t_te)
        # methods['TEASER++']['all'].append({
        #     'pair': f"{ci}-{ci+1}", 's': float(s_te), 'time': teaser_time,
        #     'rot_angle': rotation_angle(R_te), **teaser_res,
        # })
        # methods['TEASER++']['sim3'].append((s_te, R_te, t_te))

        print(f"  IRLS:     s={s_i:.5f}  res_mean={irls_res['mean']:.6f}  res_med={irls_res['median']:.6f}  time={irls_time:.3f}s")
        print(f"  TurboReg: s={s_t:.5f}  res_mean={tr_res['mean']:.6f}  res_med={tr_res['median']:.6f}  time={tr_time:.3f}s")

    # Delete last chunk file and clean up chunks directory
    if not args.chunks_dir:
        last_chunk = os.path.join(chunks_dir, f"chunk_{num_chunks-1}.npy")
        if os.path.exists(last_chunk):
            os.remove(last_chunk)
            print(f"  Deleted {last_chunk}")
        if os.path.isdir(chunks_dir) and not os.listdir(chunks_dir):
            os.rmdir(chunks_dir)
            print(f"  Removed empty directory: {chunks_dir}")

    # ---- Summary table ----
    method_names = list(methods.keys())

    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)

    def agg(results, key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.median(vals), np.max(vals), np.std(vals)

    header = f"{'Metric':<25}" + "".join(f" {name+' (mean/med/max)':>35}" for name in method_names)
    print(header)
    print("-" * 130)

    for key, label in [('mean', 'Residual Mean'),
                       ('median', 'Residual Median'),
                       ('p90', 'Residual P90'),
                       ('s', 'Scale'),
                       ('time', 'Time (s)')]:
        row = f"  {label:<23}"
        for name in method_names:
            m, md, mx, _ = agg(methods[name]['all'], key)
            row += f" {m:>10.6f} / {md:>10.6f} / {mx:>10.6f}"
        print(row)

    print("=" * 130)

    # ---- GT evaluation (ATE + RPE) ----
    gt_ate = {}
    gt_rpe = {}
    aligned_trajectories = {}
    gt_chunk_positions = None

    if args.gt_poses:
        from loop_utils.sim3utils import accumulate_sim3_transforms

        gt_poses = load_kitti_poses(args.gt_poses)
        print(f"\nGT poses loaded: {len(gt_poses)}")

        # GT chunk center positions (use chunk midpoints)
        chunk_size = config['Model']['chunk_size']
        step = chunk_size - overlap
        gt_chunk_positions = []
        gt_chunk_rotations = []
        for ci in range(num_chunks):
            frame_idx = ci * step + chunk_size // 2
            if frame_idx < len(gt_poses):
                gt_chunk_positions.append(gt_poses[frame_idx][:3, 3])
                gt_chunk_rotations.append(gt_poses[frame_idx][:3, :3])
        gt_chunk_positions = np.array(gt_chunk_positions)
        gt_chunk_rotations = np.array(gt_chunk_rotations)

        for method_name in method_names:
            sim3_list = methods[method_name]['sim3']
            cum = accumulate_sim3_transforms(sim3_list)
            positions = [np.zeros(3)]
            rotations = [np.eye(3)]
            for s, R, t in cum:
                positions.append(t.copy())
                rotations.append(R.copy())
            positions = np.array(positions[:len(gt_chunk_positions)])
            rotations = rotations[:len(gt_chunk_positions)]

            if len(positions) == len(gt_chunk_positions):
                # --- ATE ---
                ate, aligned = compute_ate(positions, gt_chunk_positions)
                gt_ate[method_name] = ate
                aligned_trajectories[method_name] = aligned

                print(f"\n  [{method_name}] ATE: RMSE={ate['rmse']:.4f}m  "
                      f"Mean={ate['mean']:.4f}m  Median={ate['median']:.4f}m  Max={ate['max']:.4f}m")

                # --- RPE (Relative Pose Error) ---
                # Compute RPE at delta=1 (consecutive chunks)
                n_pos = len(positions)
                rpe_trans_errors = []
                rpe_rot_errors = []
                for i in range(n_pos - 1):
                    # Predicted relative
                    pred_rel_t = positions[i+1] - positions[i]
                    # GT relative
                    gt_rel_t = gt_chunk_positions[i+1] - gt_chunk_positions[i]
                    # RPE translation: ||pred_rel - gt_rel||
                    rpe_trans_errors.append(float(np.linalg.norm(pred_rel_t - gt_rel_t)))
                    # RPE rotation
                    gt_rel_R = gt_chunk_rotations[i].T @ gt_chunk_rotations[i+1]
                    pred_rel_R = np.array(rotations[i]).T @ np.array(rotations[i+1])
                    rpe_rot_errors.append(rotation_diff(pred_rel_R, gt_rel_R))

                rpe = {
                    'trans_rmse': float(np.sqrt(np.mean(np.array(rpe_trans_errors)**2))),
                    'trans_mean': float(np.mean(rpe_trans_errors)),
                    'trans_median': float(np.median(rpe_trans_errors)),
                    'trans_max': float(np.max(rpe_trans_errors)),
                    'rot_rmse': float(np.sqrt(np.mean(np.array(rpe_rot_errors)**2))),
                    'rot_mean': float(np.mean(rpe_rot_errors)),
                    'rot_median': float(np.median(rpe_rot_errors)),
                    'rot_max': float(np.max(rpe_rot_errors)),
                }
                gt_rpe[method_name] = rpe
                print(f"  [{method_name}] RPE-t: RMSE={rpe['trans_rmse']:.4f}m  "
                      f"Mean={rpe['trans_mean']:.4f}m  Median={rpe['trans_median']:.4f}m")
                print(f"  [{method_name}] RPE-r: RMSE={rpe['rot_rmse']:.4f}deg  "
                      f"Mean={rpe['rot_mean']:.4f}deg  Median={rpe['rot_median']:.4f}deg")

    # ---- Paper-ready summary table ----
    col_w = 16
    total_w = 30 + col_w * len(method_names)

    # Precompute per-method stats
    method_stats = {}
    for name in method_names:
        all_results = methods[name]['all']
        scales = [r['s'] for r in all_results]
        method_stats[name] = {
            'time_mean': float(np.mean([r['time'] for r in all_results])),
            'time_total': float(np.sum([r['time'] for r in all_results])),
            'scales': scales,
        }

    irls_time_mean = method_stats['IRLS']['time_mean']

    print("\n" + "=" * total_w)
    print("PAPER-READY METRICS")
    print("=" * total_w)
    print(f"{'Metric':<30}" + "".join(f"{name:>{col_w}}" for name in method_names))
    print("-" * total_w)

    # Per-pair alignment quality
    for metric_key, metric_label in [('mean', 'Residual Mean (m)'),
                                      ('median', 'Residual Median (m)'),
                                      ('p90', 'Residual P90 (m)')]:
        row = f"  {metric_label:<28}"
        for name in method_names:
            val = float(np.mean([r[metric_key] for r in methods[name]['all']]))
            row += f"{val:>{col_w}.4f}"
        print(row)

    # Scale stability
    for stat_label, stat_fn in [('Scale Mean', lambda s: np.mean(s)),
                                 ('Scale Std', lambda s: np.std(s)),
                                 ('|Scale - 1| Mean', lambda s: np.mean(np.abs(np.array(s) - 1)))]:
        row = f"  {stat_label:<28}"
        for name in method_names:
            val = float(stat_fn(method_stats[name]['scales']))
            row += f"{val:>{col_w}.4f}"
        print(row)

    # Timing
    row = f"  {'Time Mean (s)':<28}"
    for name in method_names:
        row += f"{method_stats[name]['time_mean']:>{col_w}.4f}"
    print(row)

    row = f"  {'Time Total (s)':<28}"
    for name in method_names:
        row += f"{method_stats[name]['time_total']:>{col_w}.4f}"
    print(row)

    row = f"  {'Speedup vs IRLS':<28}"
    for name in method_names:
        t = method_stats[name]['time_mean']
        sp = irls_time_mean / t if t > 0 else float('inf')
        row += f"{f'{sp:.1f}x':>{col_w}}"
    print(row)

    # GT-based metrics
    if gt_ate:
        print("-" * total_w)
        for metric_key, metric_label in [('rmse', 'ATE RMSE (m)'), ('mean', 'ATE Mean (m)'),
                                          ('median', 'ATE Median (m)'), ('max', 'ATE Max (m)')]:
            row = f"  {metric_label:<28}"
            for name in method_names:
                val = gt_ate.get(name, {}).get(metric_key, float('nan'))
                row += f"{val:>{col_w}.4f}"
            print(row)

    if gt_rpe:
        print("-" * total_w)
        for metric_key, metric_label in [('trans_rmse', 'RPE-t RMSE (m)'), ('trans_mean', 'RPE-t Mean (m)'),
                                          ('trans_median', 'RPE-t Median (m)'),
                                          ('rot_rmse', 'RPE-r RMSE (deg)'), ('rot_mean', 'RPE-r Mean (deg)'),
                                          ('rot_median', 'RPE-r Median (deg)')]:
            row = f"  {metric_label:<28}"
            for name in method_names:
                val = gt_rpe.get(name, {}).get(metric_key, float('nan'))
                row += f"{val:>{col_w}.4f}"
            print(row)

    print("=" * total_w)

    # ---- Save results ----
    results = {
        'config': {
            'overlap': overlap,
            'chunk_size': config['Model']['chunk_size'],
            'using_sim3': config['Model']['using_sim3'],
        },
        'per_pair': {name.lower(): methods[name]['all'] for name in method_names},
        'summary': {},
    }
    for name in method_names:
        all_results = methods[name]['all']
        scales = method_stats[name]['scales']
        results['summary'][name.lower()] = {
            'residual_mean': float(np.mean([r['mean'] for r in all_results])),
            'residual_median': float(np.median([r['median'] for r in all_results])),
            'residual_p90': float(np.mean([r['p90'] for r in all_results])),
            'scale_mean': float(np.mean(scales)),
            'scale_std': float(np.std(scales)),
            'time_mean': method_stats[name]['time_mean'],
            'time_total': method_stats[name]['time_total'],
            'speedup_vs_irls': irls_time_mean / method_stats[name]['time_mean'] if method_stats[name]['time_mean'] > 0 else 0,
        }
    if gt_ate:
        results['ate'] = gt_ate
    if gt_rpe:
        results['rpe'] = gt_rpe

    json_path = os.path.join(args.output_dir, 'alignment_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ---- Plots ----
    n_pairs = len(methods['IRLS']['all'])
    x = np.arange(n_pairs)
    n_methods = len(method_names)
    bar_w = 0.8 / n_methods
    colors = {'IRLS': 'tab:blue', 'TurboReg': 'tab:red'}  # 'TEASER++': 'tab:green' temporarily disabled
    markers = {'IRLS': 'o', 'TurboReg': 's'}  # 'TEASER++': '^' temporarily disabled

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residual mean
    for i, name in enumerate(method_names):
        offset = (i - (n_methods - 1) / 2) * bar_w
        axes[0, 0].bar(x + offset, [r['mean'] for r in methods[name]['all']], bar_w,
                        label=name, color=colors[name], alpha=0.8)
    axes[0, 0].set_ylabel('Residual (mean)')
    axes[0, 0].set_title('Per-Pair Mean Residual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Residual p90
    for i, name in enumerate(method_names):
        offset = (i - (n_methods - 1) / 2) * bar_w
        axes[0, 1].bar(x + offset, [r['p90'] for r in methods[name]['all']], bar_w,
                        label=name, color=colors[name], alpha=0.8)
    axes[0, 1].set_ylabel('Residual (P90)')
    axes[0, 1].set_title('Per-Pair P90 Residual')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Scale
    for name in method_names:
        axes[1, 0].plot(x, [r['s'] for r in methods[name]['all']],
                        f'{markers[name]}-', label=name, color=colors[name])
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_ylabel('Scale')
    axes[1, 0].set_title('Per-Pair Estimated Scale')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Time
    for i, name in enumerate(method_names):
        offset = (i - (n_methods - 1) / 2) * bar_w
        axes[1, 1].bar(x + offset, [r['time'] for r in methods[name]['all']], bar_w,
                        label=name, color=colors[name], alpha=0.8)
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_title('Per-Pair Alignment Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Pair Index')

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'alignment_comparison.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {fig_path}")

    # ---- Trajectory plot (if GT available) ----
    if aligned_trajectories and gt_chunk_positions is not None:
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        # Use XZ plane (top-down view for driving datasets)
        ax2.plot(gt_chunk_positions[:, 0], gt_chunk_positions[:, 2],
                 'k-o', label='Ground Truth', markersize=4, linewidth=2)
        traj_colors = {'IRLS': 'tab:blue', 'TurboReg': 'tab:red'}  # 'TEASER++': 'tab:green' temporarily disabled
        for method_name, aligned in aligned_trajectories.items():
            ax2.plot(aligned[:, 0], aligned[:, 2],
                     '--s', label=method_name, color=traj_colors.get(method_name, 'tab:orange'),
                     markersize=4, linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Trajectory Comparison (Top-Down View)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        plt.tight_layout()
        traj_path = os.path.join(args.output_dir, 'trajectory_comparison.png')
        plt.savefig(traj_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Trajectory plot saved to {traj_path}")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

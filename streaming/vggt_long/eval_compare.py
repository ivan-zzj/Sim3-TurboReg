"""
Evaluate and compare VGGT-Long alignment methods: IRLS vs TurboReg+Umeyama.

This script compares the two alignment methods on the same VGGT chunk data
by measuring:
  1. Pairwise alignment quality (RE, TE, SE between adjacent chunks)
  2. Camera pose accuracy vs ground truth (ATE, RPE)
  3. Per-chunk alignment residuals

Usage:
    python eval_compare.py \
        --gt_poses /path/to/gt_poses.txt \
        --irls_dir /path/to/irls_exp \
        --turboreg_dir /path/to/turboreg_exp \
        --output_dir /path/to/eval_output

    # Or evaluate a single method:
    python eval_compare.py \
        --gt_poses /path/to/gt_poses.txt \
        --irls_dir /path/to/irls_exp \
        --output_dir /path/to/eval_output

    # KITTI format GT poses (12 floats per line):
    python eval_compare.py \
        --gt_poses /path/to/poses/00.txt --gt_format kitti \
        --irls_dir /path/to/irls_exp \
        --turboreg_dir /path/to/turboreg_exp \
        --output_dir /path/to/eval_output
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def load_poses_vggtlong(path):
    """Load camera_poses.txt (16 floats per line -> 4x4 C2W)."""
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
            elif len(vals) == 12:
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return poses


def load_poses_kitti(path):
    """Load KITTI format poses (12 floats per line -> 4x4)."""
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def load_poses(path, fmt='auto'):
    """Load poses, auto-detecting format."""
    if fmt == 'kitti':
        return load_poses_kitti(path)
    if fmt == 'vggtlong':
        return load_poses_vggtlong(path)
    # Auto-detect: check first line
    with open(path, 'r') as f:
        first_line = f.readline().strip().split()
    if len(first_line) == 12:
        return load_poses_kitti(path)
    return load_poses_vggtlong(path)


# ---------------------------------------------------------------------------
# Alignment: Umeyama for aligning predicted trajectory to GT
# ---------------------------------------------------------------------------

def umeyama_alignment(src, dst, with_scale=True):
    """Align src to dst using Umeyama. Returns (s, R, t)."""
    n = src.shape[0]
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    X = src - mu_s
    Y = dst - mu_d
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = float(np.mean(np.sum(X ** 2, axis=1)))
        if var_s < 1e-12:
            scale = 1.0
        else:
            scale = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ate(pred_poses, gt_poses, align=True):
    """
    Absolute Trajectory Error (ATE).
    Returns: (ate_rmse, ate_mean, ate_median, ate_std, aligned_pred_positions)
    """
    n = min(len(pred_poses), len(gt_poses))
    pred_positions = np.array([p[:3, 3] for p in pred_poses[:n]])
    gt_positions = np.array([p[:3, 3] for p in gt_poses[:n]])

    if align:
        s, R, t = umeyama_alignment(pred_positions, gt_positions, with_scale=True)
        aligned = s * (pred_positions @ R.T) + t
    else:
        aligned = pred_positions

    errors = np.linalg.norm(aligned - gt_positions, axis=1)
    return {
        'ate_rmse': float(np.sqrt(np.mean(errors ** 2))),
        'ate_mean': float(np.mean(errors)),
        'ate_median': float(np.median(errors)),
        'ate_std': float(np.std(errors)),
        'ate_max': float(np.max(errors)),
        'errors': errors,
        'aligned_positions': aligned,
        'gt_positions': gt_positions,
    }


def rotation_error(R1, R2):
    """Rotation error in degrees between two 3x3 rotation matrices."""
    R_diff = R1 @ R2.T
    cos_angle = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_rpe(pred_poses, gt_poses, delta=1):
    """
    Relative Pose Error (RPE) with given frame delta.
    Returns: dict with rpe_trans and rpe_rot statistics.
    """
    n = min(len(pred_poses), len(gt_poses))
    trans_errors = []
    rot_errors = []

    for i in range(n - delta):
        # GT relative
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        # Pred relative
        pred_rel = np.linalg.inv(pred_poses[i]) @ pred_poses[i + delta]
        # Error
        err = np.linalg.inv(gt_rel) @ pred_rel
        trans_errors.append(np.linalg.norm(err[:3, 3]))
        rot_errors.append(rotation_error(err[:3, :3], np.eye(3)))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    return {
        'rpe_trans_rmse': float(np.sqrt(np.mean(trans_errors ** 2))),
        'rpe_trans_mean': float(np.mean(trans_errors)),
        'rpe_trans_median': float(np.median(trans_errors)),
        'rpe_rot_rmse': float(np.sqrt(np.mean(rot_errors ** 2))),
        'rpe_rot_mean': float(np.mean(rot_errors)),
        'rpe_rot_median': float(np.median(rot_errors)),
        'trans_errors': trans_errors,
        'rot_errors': rot_errors,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_trajectories(results, output_path, title='Trajectory Comparison'):
    """Plot 2D trajectories (top-down view) for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if 'gt' in results:
        gt_pos = results['gt']['gt_positions']
        ax.plot(gt_pos[:, 0], gt_pos[:, 2], 'k-', linewidth=2, label='Ground Truth')

    colors = {'irls': 'tab:blue', 'turboreg': 'tab:red'}
    for method_name, data in results.items():
        if method_name == 'gt':
            continue
        pos = data['aligned_positions']
        c = colors.get(method_name, 'tab:green')
        ax.plot(pos[:, 0], pos[:, 2], '-', color=c, linewidth=1.5,
                alpha=0.8, label=f'{method_name} (ATE={data["ate_rmse"]:.3f}m)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plot saved to {output_path}")


def plot_ate_over_time(results, output_path, title='ATE Over Time'):
    """Plot ATE error for each frame."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    colors = {'irls': 'tab:blue', 'turboreg': 'tab:red'}
    for method_name, data in results.items():
        if method_name == 'gt':
            continue
        errors = data['errors']
        c = colors.get(method_name, 'tab:green')
        ax.plot(errors, '-', color=c, linewidth=1, alpha=0.8, label=f'{method_name}')

    ax.set_xlabel('Frame Index')
    ax.set_ylabel('ATE (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"ATE plot saved to {output_path}")


def plot_rpe_over_time(results, output_path, title='RPE Over Time'):
    """Plot RPE trans and rot errors."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = {'irls': 'tab:blue', 'turboreg': 'tab:red'}
    for method_name, data in results.items():
        if method_name == 'gt':
            continue
        c = colors.get(method_name, 'tab:green')
        rpe = data.get('rpe', {})
        if 'trans_errors' in rpe:
            ax1.plot(rpe['trans_errors'], '-', color=c, linewidth=0.8, alpha=0.8, label=f'{method_name}')
        if 'rot_errors' in rpe:
            ax2.plot(rpe['rot_errors'], '-', color=c, linewidth=0.8, alpha=0.8, label=f'{method_name}')

    ax1.set_ylabel('RPE Translation (m)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('RPE Rotation (deg)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"RPE plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def load_timing(exp_dir):
    """Load timing.json from experiment directory."""
    timing_path = os.path.join(exp_dir, 'timing.json')
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            return json.load(f)
    return None


def evaluate_method(method_name, exp_dir, gt_poses):
    """Evaluate a single method's output."""
    poses_path = os.path.join(exp_dir, 'camera_poses.txt')
    if not os.path.exists(poses_path):
        print(f"[{method_name}] camera_poses.txt not found in {exp_dir}")
        return None

    pred_poses = load_poses(poses_path, fmt='vggtlong')
    print(f"[{method_name}] Loaded {len(pred_poses)} predicted poses")

    n = min(len(pred_poses), len(gt_poses))
    if n < 2:
        print(f"[{method_name}] Too few poses for evaluation")
        return None

    # ATE
    ate = compute_ate(pred_poses[:n], gt_poses[:n], align=True)

    # RPE (delta=1)
    rpe = compute_rpe(pred_poses[:n], gt_poses[:n], delta=1)

    # RPE (delta=10)
    rpe10 = compute_rpe(pred_poses[:n], gt_poses[:n], delta=10)

    # Load timing
    timing = load_timing(exp_dir)

    result = {
        'method': method_name,
        'num_poses': n,
        'ate_rmse': ate['ate_rmse'],
        'ate_mean': ate['ate_mean'],
        'ate_median': ate['ate_median'],
        'ate_std': ate['ate_std'],
        'ate_max': ate['ate_max'],
        'rpe1_trans_rmse': rpe['rpe_trans_rmse'],
        'rpe1_trans_mean': rpe['rpe_trans_mean'],
        'rpe1_rot_rmse': rpe['rpe_rot_rmse'],
        'rpe1_rot_mean': rpe['rpe_rot_mean'],
        'rpe10_trans_rmse': rpe10['rpe_trans_rmse'],
        'rpe10_trans_mean': rpe10['rpe_trans_mean'],
        'rpe10_rot_rmse': rpe10['rpe_rot_rmse'],
        'rpe10_rot_mean': rpe10['rpe_rot_mean'],
        # Keep arrays for plotting
        'errors': ate['errors'],
        'aligned_positions': ate['aligned_positions'],
        'gt_positions': ate['gt_positions'],
        'rpe': rpe,
    }

    if timing:
        result['inference_time'] = timing.get('inference_time')
        result['alignment_time'] = timing.get('alignment_time')
        result['total_time'] = (timing.get('inference_time', 0) + timing.get('alignment_time', 0))

    return result


def print_comparison_table(results):
    """Print a comparison table."""
    has_timing = any(data.get('alignment_time') is not None
                     for name, data in results.items()
                     if name != 'gt' and data is not None)

    print("\n" + "=" * 120)
    print("EVALUATION RESULTS")
    print("=" * 120)

    if has_timing:
        header = (f"{'Method':<15} {'ATE_RMSE':>10} {'ATE_Mean':>10} {'ATE_Med':>10} "
                  f"{'RPE1_T':>10} {'RPE1_R':>10} {'RPE10_T':>10} {'RPE10_R':>10} "
                  f"{'Align(s)':>10} {'Total(s)':>10}")
    else:
        header = (f"{'Method':<15} {'ATE_RMSE':>10} {'ATE_Mean':>10} {'ATE_Med':>10} "
                  f"{'RPE1_T':>10} {'RPE1_R':>10} {'RPE10_T':>10} {'RPE10_R':>10}")
    print(header)
    print("-" * 120)

    for name, data in results.items():
        if name == 'gt' or data is None:
            continue
        row = (f"{name:<15} "
               f"{data['ate_rmse']:>10.4f} "
               f"{data['ate_mean']:>10.4f} "
               f"{data['ate_median']:>10.4f} "
               f"{data['rpe1_trans_rmse']:>10.4f} "
               f"{data['rpe1_rot_rmse']:>10.4f} "
               f"{data['rpe10_trans_rmse']:>10.4f} "
               f"{data['rpe10_rot_rmse']:>10.4f}")
        if has_timing:
            align_t = data.get('alignment_time')
            total_t = data.get('total_time')
            row += f" {align_t:>10.2f}" if align_t is not None else f" {'N/A':>10}"
            row += f" {total_t:>10.2f}" if total_t is not None else f" {'N/A':>10}"
        print(row)

    print("=" * 120)
    print("Units: ATE/RPE_T in meters, RPE_R in degrees, Time in seconds")

    # Print speedup if both methods have timing
    method_data = {name: data for name, data in results.items() if name != 'gt' and data is not None}
    if has_timing and 'irls' in method_data and 'turboreg' in method_data:
        irls_align = method_data['irls'].get('alignment_time', 0)
        tr_align = method_data['turboreg'].get('alignment_time', 0)
        if tr_align > 0:
            print(f"\nAlignment speedup: {irls_align/tr_align:.1f}x faster (IRLS: {irls_align:.2f}s vs TurboReg: {tr_align:.2f}s)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare VGGT-Long alignment methods')
    parser.add_argument('--gt_poses', type=str, required=True,
                        help='Path to ground truth poses file')
    parser.add_argument('--gt_format', type=str, default='auto', choices=['auto', 'kitti', 'vggtlong'],
                        help='Ground truth pose format')
    parser.add_argument('--irls_dir', type=str, default=None,
                        help='Path to IRLS experiment output directory')
    parser.add_argument('--turboreg_dir', type=str, default=None,
                        help='Path to TurboReg experiment output directory')
    parser.add_argument('--output_dir', type=str, default='/path/to/eval_results',
                        help='Path to save evaluation results')
    parser.add_argument('--rpe_delta', type=int, nargs='+', default=[1, 10],
                        help='RPE frame deltas to evaluate')
    args = parser.parse_args()

    if args.irls_dir is None and args.turboreg_dir is None:
        print("Error: At least one of --irls_dir or --turboreg_dir must be provided")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Load GT
    print(f"Loading GT poses from {args.gt_poses}...")
    gt_poses = load_poses(args.gt_poses, fmt=args.gt_format)
    print(f"Loaded {len(gt_poses)} GT poses")

    results = {}

    if args.irls_dir:
        print(f"\nEvaluating IRLS method from {args.irls_dir}...")
        results['irls'] = evaluate_method('irls', args.irls_dir, gt_poses)

    if args.turboreg_dir:
        print(f"\nEvaluating TurboReg method from {args.turboreg_dir}...")
        results['turboreg'] = evaluate_method('turboreg', args.turboreg_dir, gt_poses)

    # Print comparison
    print_comparison_table(results)

    # Save numeric results as JSON
    json_results = {}
    for name, data in results.items():
        if data is None:
            continue
        json_results[name] = {k: v for k, v in data.items()
                              if not isinstance(v, np.ndarray)}
        # Remove rpe sub-dict with arrays
        if 'rpe' in json_results[name]:
            rpe_clean = {k: v for k, v in json_results[name]['rpe'].items()
                        if not isinstance(v, np.ndarray)}
            json_results[name]['rpe'] = rpe_clean

    json_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Numeric results saved to {json_path}")

    # Store GT positions for plotting
    for name, data in results.items():
        if data is not None:
            results['gt'] = {'gt_positions': data['gt_positions']}
            break

    # Plots
    plot_trajectories(results, os.path.join(args.output_dir, 'trajectory_comparison.png'))
    plot_ate_over_time(results, os.path.join(args.output_dir, 'ate_over_time.png'))
    plot_rpe_over_time(results, os.path.join(args.output_dir, 'rpe_over_time.png'))

    print(f"\nAll results saved to {args.output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

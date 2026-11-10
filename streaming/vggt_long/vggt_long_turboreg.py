"""
VGGT-Long with TurboReg+Umeyama alignment (replacing IRLS).

Instead of using robust weighted IRLS for chunk alignment, this pipeline:
1. Uses confidence filtering to select high-quality 3D correspondences
2. Feeds correspondences to TurboReg (ScaleTurboRegGPU) for robust SIM(3) estimation
3. Everything else (chunking, loop closure, point cloud saving) remains the same

Usage:
    python vggt_long_turboreg.py --image_dir <path> [--config configs/base_config.yaml]

Requires:
    - TurboReg Python bindings (turboreg_gpu) installed
    - Same dependencies as vggt_long.py
"""

import numpy as np
import argparse
import os
import glob
import torch
from tqdm.auto import tqdm
import cv2
import gc
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_models_path = os.path.join(current_dir, 'base_models')
if base_models_path not in sys.path:
    sys.path.append(base_models_path)

# Add TurboReg bindings to path
TURBOREG_BINDINGS_PATH = os.environ.get(
    'TURBOREG_BINDINGS',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bindings'))
if TURBOREG_BINDINGS_PATH not in sys.path:
    sys.path.insert(0, TURBOREG_BINDINGS_PATH)

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
from base_models.base_model import VGGTAdapter, Pi3Adapter, MapAnythingAdapter

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    apply_sim3_direct,
    save_confident_pointcloud_batch,
    merge_ply_files,
    process_loop_list,
    compute_sim3_ab,
    compute_alignment_error,
)
from datetime import datetime
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loop_utils.config_utils import load_config
from pathlib import Path


def _init_turboreg(config):
    """Initialize ScaleTurboRegGPU from config."""
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
    return reger


def _weighted_umeyama_sim3(src, tgt, w):
    """Confidence-weighted Umeyama: tgt ~= s * R @ src + t (with scale)."""
    w = w.astype(np.float64)
    src = src.astype(np.float64)
    tgt = tgt.astype(np.float64)
    W = w.sum()
    mu_s = (w[:, None] * src).sum(axis=0) / W
    mu_t = (w[:, None] * tgt).sum(axis=0) / W
    X = src - mu_s
    Y = tgt - mu_t
    cov = (Y * w[:, None]).T @ X / W
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (w * (X ** 2).sum(axis=1)).sum() / W
    s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    if not np.isfinite(s) or s <= 1e-6:
        s = 1.0
    t = mu_t - s * (R @ mu_s)
    return s, R, t


def turboreg_align_point_maps(point_map1, conf1, point_map2, conf2, mask,
                               conf_threshold, reger, device, config):
    """
    Align point_map2 -> point_map1 using TurboReg+Umeyama instead of IRLS.

    1. Filter by confidence to get correspondences
    2. Feed to ScaleTurboRegGPU
    3. Extract (s, R, t) from the returned 4x4 Sim(3) matrix

    Returns: (s, R, t) tuple
    """
    b1 = point_map1.shape[0]
    b2 = point_map2.shape[0]
    b = min(b1, b2)

    aligned_points1 = []
    aligned_points2 = []
    confidence_weights = []

    for i in range(b):
        mask1 = conf1[i] > conf_threshold
        mask2 = conf2[i] > conf_threshold
        valid_mask = mask1 & mask2
        if mask is not None:
            valid_mask = valid_mask & mask[i].squeeze()

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        pts1 = point_map1[i][idx]
        pts2 = point_map2[i][idx]
        combined_conf = np.sqrt(conf1[i][idx] * conf2[i][idx])

        # Filter non-finite points
        ok = np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(axis=1)
        pts1 = pts1[ok]
        pts2 = pts2[ok]
        combined_conf = combined_conf[ok]

        aligned_points1.append(pts1)
        aligned_points2.append(pts2)
        confidence_weights.append(combined_conf)

    if len(aligned_points1) == 0:
        raise ValueError("No matching point pairs were found!")

    all_pts1 = np.concatenate(aligned_points1, axis=0)  # target
    all_pts2 = np.concatenate(aligned_points2, axis=0)  # source
    all_weights = np.concatenate(confidence_weights, axis=0)

    print(f"[TurboReg] Correspondences: {all_pts1.shape[0]}")

    full_pts1, full_pts2, full_weights = all_pts1, all_pts2, all_weights

    # Confidence-weighted subsampling
    max_n = config.get('TurboReg', {}).get('max_N', 7000)
    if all_pts1.shape[0] > max_n:
        prob = all_weights / all_weights.sum()
        idx = np.random.choice(all_pts1.shape[0], size=max_n, replace=False, p=prob)
        all_pts1 = all_pts1[idx]
        all_pts2 = all_pts2[idx]
        all_weights = all_weights[idx]
        print(f"[TurboReg] Confidence-weighted subsampled to {max_n} correspondences")

    # Convert to torch tensors and move to device
    kpts_src = torch.from_numpy(all_pts2.astype(np.float32)).to(device)
    kpts_tgt = torch.from_numpy(all_pts1.astype(np.float32)).to(device)

    # Run TurboReg registration
    trans_pred = reger.run_reg(kpts_src, kpts_tgt)
    trans_np = trans_pred.cpu().numpy()

    # Extract (s, R, t) from 4x4 Sim(3) matrix [[sR, t], [0, 0, 0, 1]]
    sR = trans_np[:3, :3]
    t = trans_np[:3, 3]
    s = np.linalg.norm(sR[:, 0])
    if s > 1e-6:
        R = sR / s
    else:
        s = 1.0
        R = np.eye(3, dtype=np.float32)

    # Ensure proper rotation matrix (closest rotation via SVD)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Degenerate-edge guard: adjacent-chunk Sim(3) must stay in a plausible
    # range; if TurboReg latches onto a spurious clique, fall back to
    # confidence-weighted Umeyama on the full correspondence set.
    tr_cfg = config.get('TurboReg', {})
    max_rot_rad = np.deg2rad(tr_cfg.get('fallback_max_rot_deg', 30.0))
    max_logs = tr_cfg.get('fallback_max_logscale', 0.3)
    rot_angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(np.log(max(s, 1e-8))) > max_logs or rot_angle > max_rot_rad:
        print(f'[TurboReg] Implausible edge estimate (s={s:.4f}, '
              f'rot={np.rad2deg(rot_angle):.1f} deg); falling back to '
              f'weighted Umeyama on {full_pts1.shape[0]} correspondences')
        s, R, t = _weighted_umeyama_sim3(full_pts2, full_pts1, full_weights)

    mean_error = compute_alignment_error(
        point_map1, conf1,
        point_map2, conf2,
        conf_threshold,
        s, R, t
    )
    print(f'[TurboReg] Mean alignment error: {mean_error}')

    return s, R, t


def remove_duplicates(data_list):
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    return result


def extract_p2_k_matrix(calib_path):
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P2:'):
                values = line.split(':')[1].split()
                values = [float(v) for v in values]
                p2_matrix = np.array(values).reshape(3, 4)
                k_matrix = p2_matrix[:3, :3]
                return k_matrix, p2_matrix
    raise ValueError("P2 not found in calibration file")


class VGGT_Long_TurboReg:
    """VGGT-Long pipeline with TurboReg+Umeyama replacing IRLS for chunk alignment."""

    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        self.delete_temp_files = self.config['Model']['delete_temp_files']

        if self.config['Weights']['model'] == 'VGGT':
            self.model = VGGTAdapter(self.config)
        elif self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        elif self.config['Weights']['model'] == 'Mapanything':
            self.model = MapAnythingAdapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}.")

        self.skyseg_session = None
        self.chunk_indices = None
        self.loop_list = []
        self.loop_optimizer = Sim3LoopOptimizer(self.config)
        self.sim3_list = []
        self.loop_sim3_list = []
        self.loop_predict_list = []
        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        # Initialize TurboReg
        self.turboreg_device = torch.device(self.device)
        self.turboreg_reger = _init_turboreg(self.config)

        print('init done (TurboReg mode).')

    def get_loop_pairs(self):
        if self.useDBoW:
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)
                frame = image_ori
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(
                    thresh=self.config['Loop']['DBoW']['thresh'],
                    num_repeat=self.config['Loop']['DBoW']['num_repeat'])
                if cands is not None:
                    (i, j) = cands
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)
                self.retrieval.save_up_to(frame_id)
        else:
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        predictions = self.model.infer_chunk(chunk_image_paths)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        if predictions['depth'] is not None:
            predictions['depth'] = np.squeeze(predictions['depth'])
        np.save(save_path, predictions)

        return predictions if is_loop or range_2 is not None else None

    def _align_chunks(self, point_map1, conf1, point_map2, conf2, mask):
        """Align two chunks using TurboReg."""
        if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
        else:
            conf_threshold = -1.0

        s, R, t = turboreg_align_point_maps(
            point_map1, conf1, point_map2, conf2, mask,
            conf_threshold=conf_threshold,
            reger=self.turboreg_reger,
            device=self.turboreg_device,
            config=self.config
        )
        return s, R, t

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        import time as _time
        self._t_inference_start = _time.time()

        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(self.chunk_indices,
                                             self.loop_list,
                                             half_window=int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)
                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)
        print(f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        del self.model
        torch.cuda.empty_cache()

        self._t_inference = _time.time() - self._t_inference_start
        self._t_align_start = _time.time()

        print("Aligning all the chunks (TurboReg mode)...")
        for chunk_idx in range(len(self.chunk_indices)-1):
            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()

            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            mask = None
            if chunk_data1["mask"] is not None:
                mask1 = chunk_data1["mask"][-self.overlap:]
                mask2 = chunk_data2["mask"][:self.overlap]
                mask = mask1.squeeze() & mask2.squeeze()

            s, R, t = self._align_chunks(point_map1, conf1, point_map2, conf2, mask)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)
            self.sim3_list.append((s, R, t))

        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                # Align chunk_a
                print('chunk_a align')
                point_map_loop = item[1]['world_points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['world_points_conf'][:chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]

                mask_a = None
                if item[1]['mask'] is not None:
                    mask_loop_a = item[1]['mask'][:chunk_a_range[1] - chunk_a_range[0]]
                    mask_chunk_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]
                    mask_a = mask_loop_a.squeeze() & mask_chunk_a.squeeze()

                s_a, R_a, t_a = self._align_chunks(point_map_a, conf_a, point_map_loop, conf_loop, mask_a)
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                # Align chunk_b
                print('chunk_b align')
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]

                mask_b = None
                if item[1]['mask'] is not None:
                    mask_loop_b = item[1]['mask'][-chunk_b_range[1] + chunk_b_range[0]:]
                    mask_chunk_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]
                    mask_b = mask_loop_b.squeeze() & mask_chunk_b.squeeze()

                s_b, R_b, t_b = self._align_chunks(point_map_b, conf_b, point_map_loop, conf_loop, mask_b)
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print('a -> b SIM 3')
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]

            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization (TurboReg)")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print('Apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f'Applying {chunk_idx + 1} -> {chunk_idx} (Total {len(self.chunk_indices) - 1})')
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"),
                                 allow_pickle=True).item()
            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy")
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"),
                                           allow_pickle=True).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)

                points_first = chunk_data_first['world_points'].reshape(-1, 3)
                colors_first = (chunk_data_first['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
                confs_first = chunk_data_first['world_points_conf'].reshape(-1)
                ply_path_first = os.path.join(self.pcd_dir, f'0_pcd.ply')
                save_confident_pointcloud_batch(
                    points=points_first,
                    colors=colors_first,
                    confs=confs_first,
                    output_path=ply_path_first,
                    conf_threshold=(np.mean(confs_first) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
                        if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True) else -1.0),
                    sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
                )

            aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"),
                                         allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first

            points = aligned_chunk_data['world_points'].reshape(-1, 3)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = aligned_chunk_data['world_points_conf'].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx + 1}_pcd.ply')
            save_confident_pointcloud_batch(
                points=points,
                colors=colors,
                confs=confs,
                output_path=ply_path,
                conf_threshold=(np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
                    if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True) else -1.0),
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        self.save_camera_poses()

        self._t_align = _time.time() - self._t_align_start
        print(f'[Timing] Inference: {self._t_inference:.2f}s  Alignment: {self._t_align:.2f}s')

        # Save timing.json
        timing = {
            'inference_time': self._t_inference,
            'alignment_time': self._t_align,
            'num_images': len(self.img_list),
            'num_chunks': len(self.chunk_indices),
        }
        import json
        with open(os.path.join(self.output_dir, 'timing.json'), 'w') as f:
            json.dump(timing, f, indent=2)

        print('Done.')

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                               glob.glob(os.path.join(self.img_dir, "*.png")))
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()
            if self.useDBoW:
                self.retrieval.close()
                gc.collect()
            else:
                del self.loop_detector
        torch.cuda.empty_cache()
        print('Loading model...')
        self.model.load()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k

        self.process_long_sequence()

    def save_camera_poses(self):
        chunk_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            c2w = first_chunk_extrinsics[i]
            all_poses[idx] = c2w
            if first_chunk_intrinsics is not None:
                all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[chunk_idx - 1]

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                c2w = chunk_extrinsics[i]
                transformed_c2w = S @ c2w
                transformed_c2w[:3, :3] /= s
                all_poses[idx] = transformed_c2w
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')
        print(f"Camera poses saved to {poses_path}")

        # Save image timestamps (extracted from filenames) for TUM-style evaluation
        timestamps_path = os.path.join(self.output_dir, "timestamps.txt")
        with open(timestamps_path, "w") as f:
            for img_path in self.img_list:
                fname = os.path.splitext(os.path.basename(img_path))[0]
                f.write(fname + "\n")
        print(f"Timestamps saved to {timestamps_path}")

        if all_intrinsics[0] is not None:
            intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
            with open(intrinsics_path, 'w') as f:
                for intrinsic in all_intrinsics:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f'{fx} {fy} {cx} {cy}\n')
            print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')
        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        if not self.delete_temp_files:
            return
        total_space = 0
        for d in [self.result_unaligned_dir, self.result_aligned_dir, self.result_loop_dir]:
            print(f'Deleting the temp files under {d}')
            for filename in os.listdir(d):
                file_path = os.path.join(d, filename)
                if os.path.isfile(file_path):
                    total_space += os.path.getsize(file_path)
                    os.remove(file_path)
        print('Deleting temp files done.')
        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGGT-Long with TurboReg alignment')
    parser.add_argument('--image_dir', type=str, required=True, help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/turboreg_config.yaml',
                        help='config path')
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    config = load_config(args.config)

    image_dir = args.image_dir
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = '/path/to/exps'

    save_dir = os.path.join(
        exp_dir, image_dir.replace("/", "_") + "_turboreg", current_datetime
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)

    vggt_long = VGGT_Long_TurboReg(image_dir, save_dir, config)
    vggt_long.run()
    vggt_long.close()

    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('All done.')
    sys.exit()

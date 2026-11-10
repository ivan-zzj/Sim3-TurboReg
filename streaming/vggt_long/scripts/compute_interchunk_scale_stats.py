#!/usr/bin/env python3
"""
Compute inter-chunk scale statistics from VGGT-Long experiment outputs.

The script mirrors the overlap slicing used in `vggt_long.py` and
`vggt_long_turboreg.py`. For every adjacent chunk pair it:
1. loads the raw chunk-local `world_points` and `world_points_conf`,
2. keeps only the overlap frames before backend alignment,
3. applies the same confidence threshold rule as the pipeline, and
4. estimates one Sim(3) scale value from the surviving correspondences.

Outputs:
  - pair_scales.csv          : one row per adjacent chunk pair
  - summary.json             : grouped summary statistics
  - percentiles.csv          : grouped percentiles and in-range ratios
  - percentiles.tex          : LaTeX table for the paper appendix
  - histogram.png            : histogram figure

Example:
  python3 scripts/compute_interchunk_scale_stats.py \
      --exp-dir /path/to/exps/.../2026-03-04-22-40-52 \
      --group-by experiment \
      --output-dir /tmp/scale_stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute inter-chunk scale histograms/percentiles from VGGT-Long outputs."
    )
    parser.add_argument(
        "--exp-dir",
        action="append",
        default=[],
        help="Path to a single experiment directory containing _tmp_results_unaligned.",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default=None,
        help="Scan this root recursively for experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where CSV/JSON/TEX/PNG outputs will be written.",
    )
    parser.add_argument(
        "--group-by",
        choices=["all", "dataset", "pipeline", "experiment"],
        default="dataset",
        help="How to aggregate histogram and percentile summaries.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[1, 5, 10, 25, 50, 75, 90, 95, 99],
        help="Percentiles to report.",
    )
    parser.add_argument(
        "--range-min",
        type=float,
        default=0.5,
        help="Lower bound of the target operating range.",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=1.5,
        help="Upper bound of the target operating range.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=24,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--max-correspondences",
        type=int,
        default=250000,
        help="Maximum correspondences per chunk pair. Use 0 to disable subsampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for confidence-weighted subsampling.",
    )
    return parser.parse_args()


def discover_experiment_dirs(exp_dirs: Iterable[str], exp_root: str | None) -> list[Path]:
    discovered = {
        Path(p).resolve()
        for p in exp_dirs
        if (Path(p).resolve() / "_tmp_results_unaligned").exists()
    }
    if exp_root is not None:
        root = Path(exp_root).resolve()
        for chunk0 in root.rglob("chunk_0.npy"):
            parent = chunk0.parent
            if parent.name != "_tmp_results_unaligned":
                continue
            discovered.add(parent.parent.resolve())
    return sorted(discovered)


def find_config_file(exp_dir: Path) -> Path | None:
    yaml_files = sorted(exp_dir.glob("*.yaml"))
    if not yaml_files:
        return None
    turboreg = [p for p in yaml_files if "turboreg" in p.name.lower()]
    return turboreg[0] if turboreg else yaml_files[0]


def load_config(exp_dir: Path) -> dict:
    config_path = find_config_file(exp_dir)
    if config_path is None:
        raise FileNotFoundError(f"No copied config YAML found under {exp_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_dataset(exp_dir: Path) -> str:
    name = exp_dir.as_posix().lower()
    if "kitti" in name:
        return "KITTI"
    if "tum" in name or "rgbd" in name or "freiburg" in name:
        return "TUM"
    return "Unknown"


def infer_pipeline(exp_dir: Path, config: dict) -> str:
    weights = config.get("Weights", {})
    model = str(weights.get("model", "unknown")).strip()
    if model:
        return model
    name = exp_dir.as_posix().lower()
    if "pi3" in name:
        return "Pi3"
    if "map" in name or "da3" in name:
        return "MapAnything"
    return "Unknown"


def format_experiment_name(exp_dir: Path) -> str:
    return f"{exp_dir.parent.name}/{exp_dir.name}"


def sort_chunk_files(exp_dir: Path) -> list[Path]:
    files = list((exp_dir / "_tmp_results_unaligned").glob("chunk_*.npy"))
    return sorted(files, key=lambda p: int(p.stem.split("_")[1]))


def estimate_scale_umeyama(src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    src = src_pts.astype(np.float64, copy=False)
    dst = dst_pts.astype(np.float64, copy=False)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    rms_src = math.sqrt(np.mean(np.sum(src_centered * src_centered, axis=1)))
    rms_dst = math.sqrt(np.mean(np.sum(dst_centered * dst_centered, axis=1)))
    if rms_src < 1e-12:
        raise ValueError("Degenerate source correspondences: near-zero RMS spread")
    return float(rms_dst / rms_src)


def subsample_correspondences(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    weights: np.ndarray,
    max_correspondences: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_correspondences <= 0 or pts_a.shape[0] <= max_correspondences:
        return pts_a, pts_b

    prob = np.asarray(weights, dtype=np.float64)
    prob = np.clip(prob, a_min=0.0, a_max=None)
    total = prob.sum()
    if not np.isfinite(total) or total <= 0:
        prob = None
    else:
        prob /= total

    idx = rng.choice(pts_a.shape[0], size=max_correspondences, replace=False, p=prob)
    return pts_a[idx], pts_b[idx]


def use_conf_filter(config: dict) -> bool:
    point_cfg = config.get("Model", {}).get("Pointcloud_Save", {})
    if "use_conf_filter" in point_cfg:
        return bool(point_cfg["use_conf_filter"])
    if "use_confidence_filtering" in point_cfg:
        return bool(point_cfg["use_confidence_filtering"])
    return True


def compute_pair_scale(
    chunk_a: dict,
    chunk_b: dict,
    overlap: int,
    conf_filter_enabled: bool,
    max_correspondences: int,
    rng: np.random.Generator,
) -> tuple[float, int, float]:
    point_map_a = chunk_a["world_points"][-overlap:]
    point_map_b = chunk_b["world_points"][:overlap]
    conf_a = chunk_a["world_points_conf"][-overlap:]
    conf_b = chunk_b["world_points_conf"][:overlap]

    if conf_filter_enabled:
        conf_threshold = float(min(np.median(conf_a), np.median(conf_b)) * 0.1)
    else:
        conf_threshold = -1.0

    valid_mask = (conf_a > conf_threshold) & (conf_b > conf_threshold)
    valid_mask &= np.isfinite(point_map_a).all(axis=-1)
    valid_mask &= np.isfinite(point_map_b).all(axis=-1)

    mask_a = chunk_a.get("mask")
    mask_b = chunk_b.get("mask")
    if mask_a is not None and mask_b is not None:
        overlap_mask_a = np.squeeze(mask_a[-overlap:])
        overlap_mask_b = np.squeeze(mask_b[:overlap])
        valid_mask &= overlap_mask_a & overlap_mask_b

    pts_a = point_map_a[valid_mask]
    pts_b = point_map_b[valid_mask]
    if pts_a.shape[0] == 0:
        raise ValueError("No valid overlap correspondences survived filtering")

    weights = np.sqrt(conf_a[valid_mask] * conf_b[valid_mask])
    pts_a, pts_b = subsample_correspondences(
        pts_a,
        pts_b,
        weights=weights,
        max_correspondences=max_correspondences,
        rng=rng,
    )
    scale = estimate_scale_umeyama(src_pts=pts_b, dst_pts=pts_a)
    return scale, int(pts_a.shape[0]), conf_threshold


def summarize_group(
    scales: list[float],
    percentiles: list[float],
    range_min: float,
    range_max: float,
) -> dict:
    arr = np.asarray(scales, dtype=np.float64)
    summary = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "inside_range_ratio": float(np.mean((arr >= range_min) & (arr <= range_max))),
        "inside_range_percent": float(100.0 * np.mean((arr >= range_min) & (arr <= range_max))),
    }
    for p in percentiles:
        summary[f"p{int(p):02d}"] = float(np.percentile(arr, p))
    return summary


def write_pair_csv(rows: list[dict], output_dir: Path) -> None:
    pair_csv = output_dir / "pair_scales.csv"
    fieldnames = [
        "experiment",
        "experiment_dir",
        "dataset",
        "pipeline",
        "chunk_pair",
        "scale",
        "log_scale",
        "num_correspondences",
        "conf_threshold",
    ]
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_percentiles_csv(
    grouped_rows: list[dict],
    percentiles: list[float],
    output_dir: Path,
) -> None:
    fieldnames = ["group", "count", "mean", "std", "min", "max"] + [
        f"p{int(p):02d}" for p in percentiles
    ] + ["inside_range_ratio", "inside_range_percent"]
    with (output_dir / "percentiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in grouped_rows:
            writer.writerow(row)


def write_percentiles_tex(grouped_rows: list[dict], percentiles: list[float], output_dir: Path) -> None:
    tex_path = output_dir / "percentiles.tex"
    pct_cols = " ".join(["c" for _ in percentiles])
    headers = " & ".join([f"$p_{{{int(p)}}}$" for p in percentiles])
    lines = [
        "\\begin{tabular}{l c " + pct_cols + " c}",
        "\\toprule",
        "Group & $N$ & " + headers + " & In-range (\\%) \\\\",
        "\\midrule",
    ]
    for row in grouped_rows:
        pct_values = " & ".join([f"{row[f'p{int(p):02d}']:.3f}" for p in percentiles])
        lines.append(
            f"{row['group']} & {row['count']} & {pct_values} & {row['inside_range_percent']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_histogram(
    grouped_scales: dict[str, list[float]],
    range_min: float,
    range_max: float,
    bins: int,
    output_dir: Path,
) -> None:
    groups = sorted(grouped_scales)
    all_scales = np.concatenate([np.asarray(grouped_scales[g], dtype=np.float64) for g in groups])
    span = max(all_scales.max() - all_scales.min(), 1e-3)
    hist_min = min(all_scales.min(), range_min) - 0.05 * span
    hist_max = max(all_scales.max(), range_max) + 0.05 * span

    plt.figure(figsize=(8.5, 4.8))
    for group in groups:
        plt.hist(
            grouped_scales[group],
            bins=bins,
            range=(hist_min, hist_max),
            histtype="step",
            linewidth=2.0,
            label=f"Group: {group} (N={len(grouped_scales[group])})",
        )

    plt.axvline(range_min, color="black", linestyle="--", linewidth=1.2)
    plt.axvline(range_max, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Estimated inter-chunk scale")
    plt.ylabel("Adjacent chunk pairs")
    plt.title("Inter-chunk scale distribution from raw VGGT local point clouds")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_dirs = discover_experiment_dirs(args.exp_dir, args.exp_root)
    if not exp_dirs:
        raise RuntimeError("No experiment directories found. Pass --exp-dir or --exp-root.")

    rows: list[dict] = []
    skipped_exp_dirs: list[str] = []

    for exp_dir in exp_dirs:
        config = load_config(exp_dir)
        overlap = int(config.get("Model", {}).get("overlap", 0))
        if overlap <= 0:
            raise ValueError(f"Invalid overlap={overlap} in {exp_dir}")

        dataset = infer_dataset(exp_dir)
        pipeline = infer_pipeline(exp_dir, config)
        experiment_name = format_experiment_name(exp_dir)
        conf_filter_enabled = use_conf_filter(config)
        chunk_files = sort_chunk_files(exp_dir)
        if len(chunk_files) < 2:
            skipped_exp_dirs.append(str(exp_dir))
            continue

        for idx in range(len(chunk_files) - 1):
            chunk_a = np.load(chunk_files[idx], allow_pickle=True).item()
            chunk_b = np.load(chunk_files[idx + 1], allow_pickle=True).item()
            scale, n_corr, conf_threshold = compute_pair_scale(
                chunk_a=chunk_a,
                chunk_b=chunk_b,
                overlap=overlap,
                conf_filter_enabled=conf_filter_enabled,
                max_correspondences=args.max_correspondences,
                rng=rng,
            )
            rows.append(
                {
                    "experiment": experiment_name,
                    "experiment_dir": str(exp_dir),
                    "dataset": dataset,
                    "pipeline": pipeline,
                    "chunk_pair": f"{idx}->{idx + 1}",
                    "scale": float(scale),
                    "log_scale": float(math.log(max(scale, 1e-12))),
                    "num_correspondences": n_corr,
                    "conf_threshold": conf_threshold,
                }
            )

    if not rows:
        if skipped_exp_dirs:
            skipped_msg = "\n".join(skipped_exp_dirs)
            raise RuntimeError(
                "No valid adjacent chunk pairs were processed. "
                "The following experiment directories do not contain usable "
                "`_tmp_results_unaligned/chunk_*.npy` files:\n"
                f"{skipped_msg}"
            )
        raise RuntimeError("No valid adjacent chunk pairs were processed.")

    write_pair_csv(rows, output_dir)

    grouped_scales: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if args.group_by == "all":
            key = "All"
        else:
            key = row[args.group_by]
        grouped_scales[key].append(float(row["scale"]))

    summary = {}
    grouped_rows = []
    for group in sorted(grouped_scales):
        stats = summarize_group(
            grouped_scales[group],
            percentiles=args.percentiles,
            range_min=args.range_min,
            range_max=args.range_max,
        )
        summary[group] = stats
        grouped_rows.append({"group": group, **stats})

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_percentiles_csv(grouped_rows, args.percentiles, output_dir)
    write_percentiles_tex(grouped_rows, args.percentiles, output_dir)
    plot_histogram(grouped_scales, args.range_min, args.range_max, args.bins, output_dir)

    print(f"Processed {len(rows)} adjacent chunk pairs from {len(exp_dirs)} experiments.")
    print(f"Wrote outputs to: {output_dir}")
    for row in grouped_rows:
        print(
            f"[{row['group']}] N={row['count']} "
            f"mean={row['mean']:.4f} "
            f"p50={row['p50']:.4f} "
            f"in-range={row['inside_range_percent']:.1f}%"
        )


if __name__ == "__main__":
    main()

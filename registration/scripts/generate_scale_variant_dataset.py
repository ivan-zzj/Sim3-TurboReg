#!/usr/bin/env python3
"""Generate a correspondence-level Sim(3) benchmark; no descriptor recomputation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compute_sim3_from_gt(gt, scale, scale_on):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be finite and positive')
    result = np.asarray(gt, dtype=np.float64).copy()
    if scale_on == 'source':
        result[:3, :3] /= scale
    elif scale_on == 'target':
        result[:3, :] *= scale
    else:
        raise ValueError('scale_on must be source or target')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--dataset', choices=['3DMatch', '3DLoMatch'], default='3DMatch')
    p.add_argument('--desc', choices=['fpfh', 'fcgf'], default='fpfh')
    p.add_argument('--scale_range', nargs=2, type=float, default=[0.5, 2.0])
    p.add_argument('--fixed_scale', type=float)
    p.add_argument('--scale_on', choices=['source', 'target'], default='source')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_pairs', type=int, default=-1)
    args = p.parse_args()
    low, high = args.scale_range
    if not (np.isfinite([low, high]).all() and 0 < low < high):
        p.error('scale_range must satisfy 0 < min < max')
    if args.fixed_scale is not None and not (np.isfinite(args.fixed_scale) and args.fixed_scale > 0):
        p.error('fixed_scale must be finite and positive')
    base = args.input_dir / args.dataset / f'all_{args.desc}'
    output = args.output_dir / args.dataset / f'all_{args.desc}'
    suffix = '@corr_fcgf.txt' if args.desc == 'fcgf' else '@corr.txt'
    gt_suffix = '@GTmat_fcgf.txt' if args.desc == 'fcgf' else '@GTmat.txt'
    files = sorted(base.glob('*/*' + suffix))
    if not files:
        p.error(f'No correspondence files found: {base}')
    if args.max_pairs > 0:
        files = files[:args.max_pairs]
    # A distinct output directory prevents mixing protocols or overwriting source data.
    if output.exists():
        p.error(f'Output already exists; choose a fresh output_dir: {output}')
    output.mkdir(parents=True)
    records = []
    for corr_file in files:
        pair = corr_file.name.removesuffix(suffix)
        pair_id = f'{args.dataset}/{corr_file.parent.name}/{pair}'
        token = hashlib.sha256(f'{args.seed}/{pair_id}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], 'little'))
        scale = args.fixed_scale if args.fixed_scale is not None else float(np.exp(rng.uniform(np.log(low), np.log(high))))
        corr = np.loadtxt(corr_file, ndmin=2)
        gt = np.loadtxt(corr_file.with_name(pair + gt_suffix))
        if gt.shape == (3, 4):
            gt = np.vstack([gt, [0, 0, 0, 1]])
        if corr.shape[1] != 6 or gt.shape != (4, 4) or not np.isfinite(corr).all() or not np.isfinite(gt).all():
            raise ValueError(f'Invalid correspondence/GT shape or values: {corr_file}')
        corr[:, :3] *= scale if args.scale_on == 'source' else 1
        corr[:, 3:] *= scale if args.scale_on == 'target' else 1
        out_gt = compute_sim3_from_gt(gt, scale, args.scale_on)
        scene = output / corr_file.parent.name
        scene.mkdir(exist_ok=True)
        out_corr, out_gt_file = scene / corr_file.name, scene / (pair + gt_suffix)
        np.savetxt(out_corr, corr, fmt='%.10g')
        np.savetxt(out_gt_file, out_gt, fmt='%.10g')
        records.append(dict(pair_id=pair_id, corr=str(out_corr.relative_to(output)),
                            gt=str(out_gt_file.relative_to(output)), applied_scale=scale,
                            scale_on=args.scale_on, target_unit_scale=scale if args.scale_on == 'target' else 1.0,
                            input_sha256=hashlib.sha256(corr_file.read_bytes()).hexdigest()))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(distribution='fixed' if args.fixed_scale is not None else 'log_uniform',
                    protocol='correspondence_level_sim3', num_pairs=len(records),
                    transform_convention='dst = s R src + t')
    (output / 'generation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Generated {len(records)} pairs: {output / "manifest.jsonl"}')


if __name__ == '__main__':
    main()

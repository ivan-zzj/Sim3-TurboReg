#!/usr/bin/env python3
"""Evaluate every manifest pair, including failures, under one explicit Sim(3) protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def decompose(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Expected finite homogeneous 4x4 transform')
    det = np.linalg.det(T[:3, :3])
    if det <= 0:
        raise ValueError('Sim(3) must have positive scale and proper rotation')
    s = np.cbrt(det)
    R = T[:3, :3] / s
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
        raise ValueError('Transform contains nonuniform scale or shear')
    return s, R, T[:3, 3]


def errors(pred, gt, target_unit_scale=1.0):
    sp, rp, tp = decompose(pred)
    sg, rg, tg = decompose(gt)
    if not np.isfinite(target_unit_scale) or target_unit_scale <= 0:
        raise ValueError('target_unit_scale must be positive')
    re = float(np.degrees(np.arccos(np.clip((np.trace(rp.T @ rg) - 1) / 2, -1, 1))))
    te = float(np.linalg.norm(tp - tg) / target_unit_scale)
    return dict(re_deg=re, te=te, log_scale_error=float(abs(np.log(sp / sg))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--method', choices=['turboreg', 'ransac', 'teaser', 'se3', 'external'], default='turboreg')
    p.add_argument('--predictions', type=Path, help='External JSONL: pair_id, transformation, runtime_ms')
    p.add_argument('--device', default='cpu')
    p.add_argument('--options', type=Path, help='JSON object with Sim3Options fields')
    p.add_argument('--ransac_iterations', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--re_threshold', type=float, default=15)
    p.add_argument('--te_threshold', type=float, default=0.3)
    p.add_argument('--log_scale_threshold', type=float, default=0.1)
    p.add_argument('--teaser_noise_bound', type=float, default=0.05)
    args = p.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in [args.re_threshold, args.te_threshold, args.log_scale_threshold, args.teaser_noise_bound]):
        p.error('Evaluation thresholds must be finite and positive')
    if args.threads <= 0 or args.ransac_iterations <= 0:
        p.error('threads and ransac_iterations must be positive')
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not records or len({r['pair_id'] for r in records}) != len(records):
        p.error('Manifest must be nonempty with unique pair_id values')
    if args.output.exists():
        p.error('Output directory already exists; choose a fresh output path')
    options_data = json.loads(args.options.read_text()) if args.options else {}
    predictions = {}
    if args.method == 'external':
        if not args.predictions:
            p.error('external requires --predictions')
        for line in args.predictions.read_text().splitlines():
            row = json.loads(line)
            if row['pair_id'] in predictions:
                p.error('Duplicate prediction pair_id')
            predictions[row['pair_id']] = row
        torch = None
    else:
        import torch
        import turboreg_gpu as backend
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            p.error('CUDA requested but unavailable')
        opts = backend.Sim3Options()
        for key, value in options_data.items():
            if not hasattr(opts, key):
                p.error(f'Unknown Sim3 option: {key}')
            setattr(opts, key, value)
        options_data = {key: getattr(opts, key) for key in dir(opts) if not key.startswith('_')}
        if args.method == 'teaser':
            import teaserpp_python as teaser
    def sync():
        if torch is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)
    rows = []
    args.output.mkdir(parents=True)
    output_file = args.output / 'pairs.jsonl'
    for i, record in enumerate(records):
        row = dict(pair_id=record['pair_id'], success=False, runtime_ms=None,
                   re_deg=None, te=None, log_scale_error=None)
        start = None
        try:
            gt = np.loadtxt(args.manifest.parent / record['gt'])
            decompose(gt)
            units = float(record.get('target_unit_scale', 1.0))
            if not np.isfinite(units) or units <= 0:
                raise ValueError('Invalid target_unit_scale')
            if args.method == 'external':
                entry = predictions[record['pair_id']]
                pred = np.asarray(entry['transformation'])
                row['runtime_ms'] = float(entry['runtime_ms'])
                if not np.isfinite(row['runtime_ms']) or row['runtime_ms'] < 0:
                    row['runtime_ms'] = None
                    raise ValueError('Invalid external runtime')
            else:
                corr = np.loadtxt(args.manifest.parent / record['corr'], ndmin=2)
                if corr.shape[1] != 6 or len(corr) < 3 or not np.isfinite(corr).all():
                    raise ValueError('Expected finite N>=3 by 6 correspondences')
                # The identical prefix is supplied to every estimator. No GT filtering.
                corr = corr[:opts.max_points]
                src = torch.as_tensor(corr[:, :3].copy(), dtype=torch.float32, device=device)
                dst = torch.as_tensor(corr[:, 3:].copy(), dtype=torch.float32, device=device)
                opts.inlier_threshold = options_data['inlier_threshold'] * units
                if args.method == 'turboreg':
                    solve = lambda: backend.register_sim3(src, dst, opts).transformation
                elif args.method == 'ransac':
                    solve = lambda: backend.ransac_sim3(src, dst, opts, args.ransac_iterations, args.seed).transformation
                elif args.method == 'se3':
                    rigid = backend.TurboRegGPU(opts.max_points, 0.012 * units, opts.num_pivots,
                                               opts.nms_radius, opts.inlier_threshold, opts.metric)
                    solve = lambda: rigid.run_reg(src, dst)
                else:
                    a, b = np.ascontiguousarray(corr[:, :3].T), np.ascontiguousarray(corr[:, 3:].T)
                    def solve():
                        params = teaser.RobustRegistrationSolver.Params()
                        params.estimate_scaling = True
                        params.noise_bound = args.teaser_noise_bound * units
                        params.rotation_estimation_algorithm = teaser.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
                        estimator = teaser.RobustRegistrationSolver(params)
                        estimator.solve(a, b)
                        solution = estimator.getSolution()
                        T = np.eye(4)
                        T[:3, :3] = solution.scale * solution.rotation
                        T[:3, 3] = solution.translation
                        return T
                if i == 0:
                    try:
                        solve()
                    except RuntimeError:
                        pass
                    sync()
                sync()
                start = time.perf_counter()
                prediction = solve()
                sync()
                row['runtime_ms'] = (time.perf_counter() - start) * 1000
                start = None
                pred = prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
                row['num_correspondences'] = len(corr)
            row.update(errors(pred, gt, units))
            row['success'] = (row['re_deg'] < args.re_threshold and row['te'] < args.te_threshold
                              and row['log_scale_error'] < args.log_scale_threshold)
            row['transformation'] = pred.tolist()
        except Exception as e:
            if start is not None:
                sync()
                row['runtime_ms'] = (time.perf_counter() - start) * 1000
            row['error'] = str(e)
        rows.append(row)
        with output_file.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        if (i+1) % 20 == 0:
            print(f'{i+1}/{len(records)} pairs', flush=True)
    times = [r['runtime_ms'] for r in rows if r['runtime_ms'] is not None and np.isfinite(r['runtime_ms'])]
    success = [r for r in rows if r['success']]
    summary = dict(method=args.method, total=len(rows), successes=len(success),
                   errors=sum('error' in r for r in rows), rr=sum(r['success'] for r in rows)/len(rows),
                   timed_pairs=len(times), runtime_median_ms=float(np.median(times)) if times else None,
                   runtime_p95_ms=float(np.percentile(times,95)) if times else None,
                   success_mean={k: float(np.mean([r[k] for r in success])) if success else None
                                 for k in ['re_deg','te','log_scale_error']},
                   manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                   arguments={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                   options=options_data, torch_version=torch.__version__ if torch is not None else None,
                   protocol='custom_sim3_joint_re_te_logscale; not official 3DMatch RR')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Replay the frozen paper-protocol inputs: TEASER++ CPU versus Ours CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import teaser_sim3_benchmark as b


def summarize(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    meta = json.loads((output / 'run.json').read_text())
    config = meta['config']
    rows = b.read_records(output / 'results.jsonl')
    expected = {(f'{o}-o{p:02d}-r{r:03d}', m) for o in config['objects']
                for p in config['outlier_percent'] for r in range(config['repeats']) for m in b.METHODS}
    actual = {(r['trial_id'], r['method']) for r in rows}
    assert actual <= expected
    by_trial = {}
    for row in rows:
        assert row['contract_sha256'] == meta['contract_sha256']
        by_trial.setdefault(row['trial_id'], set()).add(row['input_sha256'])
    assert all(len(v) == 1 for v in by_trial.values())
    labels = {'TEASER++': 'TEASER++ (CPU)', 'Ours': 'Ours (GPU)'}
    colors = ['#2166ac', '#d6604d']
    groups = []
    text = ['# TEASER++ CPU / Ours GPU: same frozen Sim(3) inputs', '',
            f'Completed calls: {len(rows)}/{len(expected)}. GPU: {meta["gpu"]["name"]}.', '',
            'Bunny follows Fig. 6(c) with documented reproduction assumptions; other objects are extensions. '
            'Median [Q1, Q3] uses all finite predictions, including inaccurate ones. Invalid results are counted separately. '
            'All-trial error means are +inf if any solver call fails. No RR threshold.', '',
            'Timing: CPU solver/API call versus synchronized CUDA API call. Input host-to-device copies, '
            'output retrieval, I/O and metrics excluded; internal CPU work and internal transfers included. '
            'GPU times are wall-clock API latency, not CUDA-event kernel time. Frozen CPU-validation parameters reused.', '']
    for obj in config['objects']:
        text += [f'## {obj}' + (' — primary Fig. 6(c)' if obj == 'bunny' else ' — extension'), '']
        fig, axes = plt.subplots(5, 1, figsize=(8, 13), constrained_layout=True)
        for pi, pct in enumerate(config['outlier_percent']):
            text += [f'### {pct}% outliers', '',
                     '| Method | Scale error | Rotation (deg) | Translation | Time (ms) | Invalid / observed |',
                     '|---|---:|---:|---:|---:|---:|']
            for mi, method in enumerate(b.METHODS):
                group = [r for r in rows if (r['object'], r['outlier_percent'], r['method']) == (obj, pct, method)]
                failed = sum(r['status'] != 'ok' for r in group)
                result = dict(object=obj, outlier_percent=pct, method=method, device=labels[method],
                              observed=len(group), failures=failed, missing=config['repeats'] - len(group))
                cells = []
                pos = pi + (mi - .5) * .32
                for ki, key in enumerate(b.METRICS):
                    vals = [r[key] for r in group if r[key] is not None]
                    stats = np.percentile(vals, [25, 50, 75]) if vals else [None] * 3
                    for suffix, v in zip(['q1_finite', 'median_finite', 'q3_finite'], stats):
                        result[f'{key}_{suffix}'] = None if v is None else float(v)
                    result[f'{key}_mean_all'] = ('incomplete' if len(group) != config['repeats'] else
                        '+inf' if failed and key != 'runtime_s' else float(np.mean(vals)) if vals else '+inf')
                    if vals:
                        factor = 1000 if key == 'runtime_s' else 1
                        q1, med, q3 = (v * factor for v in stats)
                        cells.append(f'{med:.5g} [{q1:.5g}, {q3:.5g}]')
                        axes[ki].boxplot([np.array(vals) * factor], positions=[pos], widths=.25,
                            showfliers=True, whis=1.5, patch_artist=True,
                            boxprops=dict(facecolor=colors[mi], alpha=.45), medianprops=dict(color='black'),
                            flierprops=dict(marker='.', markersize=3, markerfacecolor=colors[mi], markeredgecolor=colors[mi]))
                    else:
                        cells.append('undefined')
                axes[4].bar(pos, failed, width=.25, color=colors[mi])
                groups.append(result)
                text.append(f'| {labels[method]} | ' + ' | '.join(cells) + f' | {failed}/{len(group)} |')
            text += ['']
        for ax, label in zip(axes, ['Absolute scale error', 'Rotation error [deg]',
                'Translation [normalized units]', 'Synchronized API time [ms]', 'Invalid model / exception count']):
            ax.set_ylabel(label)
            ax.set_xticks(range(len(config['outlier_percent'])), config['outlier_percent'])
            ax.grid(axis='y', alpha=.25)
        for ax in axes[:4]:
            ax.set_yscale('log')
        axes[4].set_ylim(0, config['repeats'] + 1)
        axes[4].set_xlabel('Outlier rate [%]')
        axes[0].legend(handles=[Patch(facecolor=c, label=labels[m]) for c, m in zip(colors, b.METHODS)])
        fig.suptitle(f'{obj}: unknown scale, N=100, 40 trials per condition\n'
                     + ('Fig. 6(c) protocol with documented assumptions' if obj == 'bunny' else 'Extension: same Sim(3) protocol'))
        fig.savefig(output / f'{obj}_boxplots.png', dpi=170)
        fig.savefig(output / f'{obj}_boxplots.pdf')
        plt.close(fig)
    import csv
    with open(output / 'summary.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    b.atomic_json(output / 'summary.json', dict(complete=actual == expected, observed_calls=len(rows),
                  expected_calls=len(expected), same_frozen_inputs=True, groups=groups))
    (output / 'RESULTS.md').write_text('\n'.join(text) + '\n')
    print(f'Summary complete={actual == expected}: {output / "RESULTS.md"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=b.REPO / 'results/teaser_sim3_paper')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=1, help='Physical GPU index, exposed as cuda:0')
    p.add_argument('--limit-trials', type=int, help='Separate smoke output only')
    p.add_argument('--summary-only', action='store_true')
    args = p.parse_args()
    if args.source.resolve() == args.output.resolve():
        p.error('GPU output must differ from original CPU output')
    if args.summary_only:
        with b.exclusive(args.output):
            summarize(args.output)
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    source_meta = json.loads((args.source / 'run.json').read_text())
    config = dict(source_meta['config'], device='cuda:0')
    b.configure_runtime(config)
    np, torch = b.np, b.torch
    torch.cuda.init()  # fail clearly before writing result records if devices are unavailable
    torch.cuda.set_device(0)
    options = source_meta['ours']
    source_rows = b.read_records(args.source / 'results.jsonl')
    inputs = {r['trial_id']: r for r in source_rows if r['method'] == 'TEASER++'}
    expected_ids = [f'{o}-o{pct:02d}-r{r:03d}' for o in config['objects']
                    for pct in config['outlier_percent'] for r in range(config['repeats'])]
    if set(inputs) != set(expected_ids):
        raise ValueError('Original CPU run does not contain all expected trial inputs')
    assert b.digest(Path(b.__file__)) == source_meta['benchmark_sha256']
    gpu = dict(physical_index=args.gpu, visible_device='cuda:0', name=torch.cuda.get_device_name(0),
               capability=list(torch.cuda.get_device_capability(0)),
               total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
               torch_cuda_runtime=torch.version.cuda)
    contract = dict(config=config, ours=options, gpu=gpu, environment=b.environment(),
                    source_contract=source_meta['contract_sha256'],
                    source_results_sha256=b.digest(args.source / 'results.jsonl'),
                    replay_sha256=b.digest(Path(__file__)), base_runner_sha256=b.digest(Path(b.__file__)))
    contract_hash = hashlib.sha256(b.canonical(contract).encode()).hexdigest()
    with b.exclusive(args.output):
        meta_path = args.output / 'run.json'
        if meta_path.exists():
            if json.loads(meta_path.read_text())['contract_sha256'] != contract_hash:
                raise ValueError('Changed input, code, backend, device, environment or options: resume rejected')
        else:
            b.atomic_json(meta_path, dict(contract_sha256=contract_hash, **contract))
            shutil.copyfile(__file__, args.output / 'replay_source.py')
        for phase in ['before']:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            (args.output / f'gpu_{phase}.txt').write_text(result.stdout + result.stderr)
        # Correctness on an independent exact transform, followed by backend warm-up.
        src = np.random.default_rng(883102).normal(size=(100, 3)).astype(np.float32)
        gt = np.eye(4)
        gt[:3, :3] = 2.3 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        gt[:3, 3] = [.2, -.3, .1]
        sanity = dict(src=src, dst=(src @ gt[:3, :3].T + gt[:3, 3]).astype(np.float32), gt=gt)
        checks = []
        for method in b.METHODS:
            row = b.solve(method, sanity, config, options)
            if row['status'] != 'ok' or row['scale_error'] > 1e-4 or row['rotation_error_deg'] > .05 or row['translation_error'] > 1e-4:
                raise RuntimeError(f'Independent transform correctness failed: {row}')
            checks.append(row)
        for _ in range(20):
            for method in b.METHODS:
                b.solve(method, sanity, config, options)
        b.atomic_json(args.output / 'correctness.json', dict(passed=True, exact_transform_checks=checks, warmups_per_method=20))
        done = {(r['trial_id'], r['method']): r for r in b.read_records(args.output / 'results.jsonl', repair=True)}
        print(f'GPU {args.gpu}: {gpu["name"]}; {len(done)}/{len(expected_ids)*2} saved calls', flush=True)
        with open(args.output / 'results.jsonl', 'a') as log:
            for index, trial_id in enumerate(expected_ids):
                if args.limit_trials is not None and index >= args.limit_trials:
                    break
                io_start = time.perf_counter()
                with np.load(args.source / 'trials' / f'{trial_id}.npz', allow_pickle=False) as data:
                    trial = {k: data[k] for k in data.files}
                io_s = time.perf_counter() - io_start
                source = inputs[trial_id]
                input_sha = b.input_hash(trial['src'], trial['dst'])
                if input_sha != source['input_sha256'] or not np.array_equal(trial['gt'], source['gt']):
                    raise ValueError(f'Frozen input changed: {trial_id}')
                for method in (b.METHODS if source['repeat'] % 2 == 0 else b.METHODS[::-1]):
                    if (trial_id, method) in done:
                        assert done[(trial_id, method)]['input_sha256'] == input_sha
                        continue
                    torch.cuda.reset_peak_memory_stats()
                    row = b.solve(method, trial, config, options)
                    torch.cuda.synchronize()
                    row.update({k: source[k] for k in ['trial_id', 'object', 'outlier_percent', 'repeat', 'seed', 'gt', 'scope', 'split']})
                    row.update(device='cpu' if method == 'TEASER++' else 'cuda:0',
                               physical_gpu=None if method == 'TEASER++' else args.gpu,
                               io_s=io_s, contract_sha256=contract_hash,
                               cuda_peak_allocated_bytes=0 if method == 'TEASER++' else torch.cuda.max_memory_allocated())
                    log.write(b.canonical(row) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                    done[(trial_id, method)] = row
                    if len(done) % 40 == 0 or row['status'] != 'ok':
                        print(f'{len(done)}/{len(expected_ids)*2} {trial_id} {method}: {row["status"]}', flush=True)
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        (args.output / 'gpu_after.txt').write_text(result.stdout + result.stderr)
        print(f'Saved calls: {len(done)}/{len(expected_ids)*2}', flush=True)
        if args.limit_trials is None:
            summarize(args.output)


if __name__ == '__main__':
    main()

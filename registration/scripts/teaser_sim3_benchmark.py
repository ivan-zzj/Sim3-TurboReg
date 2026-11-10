#!/usr/bin/env python3
"""Paper Fig. 6(c) Sim(3) reproduction with explicit assumptions and paired inputs.

Commands: validate (independent seeds), run (frozen settings), summarize.
No registration-recall threshold or log-scale metric is used here.
"""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
METHODS = ['TEASER++', 'Ours']
METRICS = ['scale_error', 'rotation_error_deg', 'translation_error', 'runtime_s']


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def ball(rng, n, radius):
    v = rng.normal(size=(n, 3))
    return radius * v / np.linalg.norm(v, axis=1)[:, None] * rng.random((n, 1)) ** (1 / 3)


def bounded_noise(rng, n, sigma, beta):
    noise = rng.normal(0, sigma, (n, 3))
    bad = np.linalg.norm(noise, axis=1) > beta
    while bad.any():
        noise[bad] = rng.normal(0, sigma, (int(bad.sum()), 3))
        bad = np.linalg.norm(noise, axis=1) > beta
    return noise


def make_trial(vertices, config, object_index, outlier_pct, repeat, split='test'):
    from scipy.spatial.transform import Rotation
    seed = [config[f'{split}_seed'], object_index, outlier_pct, repeat]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    n = config['n_points']
    indices = rng.choice(len(vertices), n, replace=False)
    sampled = vertices[indices].astype(np.float64)
    offset = sampled.min(axis=0)
    extent = float(np.ptp(sampled, axis=0).max())
    if extent <= 0:
        raise ValueError('Degenerate source cloud')
    src = ((sampled - offset) / extent).astype(np.float32)
    scale = rng.uniform(*config['scale_range'])
    quat = rng.normal(size=4)
    rotation = Rotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
    translation = ball(rng, 1, config['translation_radius'])[0]
    noise = bounded_noise(rng, n, config['sigma'], config['beta'])
    dst = scale * src.astype(np.float64) @ rotation.T + translation + noise
    outlier_indices = rng.choice(n, n * outlier_pct // 100, replace=False)
    dst[outlier_indices] = ball(rng, len(outlier_indices), config['outlier_radius'])
    dst = dst.astype(np.float32)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = scale * rotation, translation
    return dict(src=src, dst=dst, gt=T, seed=np.array(seed, dtype=np.uint64),
                sample_indices=indices, outlier_indices=outlier_indices, noise=noise,
                normalization_offset=offset, normalization_extent=np.array(extent))


def input_hash(src, dst):
    h = hashlib.sha256()
    for x in (src, dst):
        h.update(np.asarray(x, dtype='<f4', order='C').tobytes())
    return h.hexdigest()


def decompose(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Nonfinite or invalid homogeneous transform')
    determinant = np.linalg.det(T[:3, :3])
    if determinant <= 0:
        raise ValueError('Nonpositive scale or reflected rotation')
    scale = np.cbrt(determinant)
    rotation = T[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError('Nonuniform scale or shear')
    return scale, rotation, T[:3, 3]


def errors(estimate, gt):
    s, R, t = decompose(estimate)
    sg, Rg, tg = decompose(gt)
    return dict(scale_error=float(abs(s - sg)),
                rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Rg) - 1) / 2, -1, 1)))),
                translation_error=float(np.linalg.norm(t - tg)))


def configure_runtime(config):
    # Before importing either native backend, constrain all OpenMP / BLAS runtimes.
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = str(config['threads'])
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    sys.path.insert(0, str(REPO / 'bindings'))
    if 'teaser_python_dir' in config:
        native_dir = REPO / config['teaser_python_dir']
        if not (native_dir / 'teaserpp_python/__init__.py').exists():
            raise ValueError('Build the pinned TEASER++ Release package first (docs/TEASER_SIM3_PROTOCOL.md)')
        sys.path.insert(0, str(native_dir))
    global np, torch, turboreg_gpu, teaserpp_python
    import numpy as np
    import torch
    import turboreg_gpu
    import teaserpp_python
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)


def backend_settings(config, ours):
    p = teaserpp_python.RobustRegistrationSolver.Params()
    cls = teaserpp_python.RobustRegistrationSolver
    for key, value in config['teaser'].items():
        if key == 'rotation_estimation_algorithm':
            value = getattr(cls.ROTATION_ESTIMATION_ALGORITHM, value)
        elif key == 'rotation_tim_graph':
            value = getattr(cls.INLIER_GRAPH_FORMULATION, value)
        elif key == 'inlier_selection_mode':
            value = getattr(cls.INLIER_SELECTION_MODE, value)
        setattr(p, key, value)
    if hasattr(p, 'max_clique_num_threads'):
        p.max_clique_num_threads = config['threads']
    o = turboreg_gpu.Sim3Options()
    for key, value in ours.items():
        setattr(o, key, value)
    return p, o


def solve(method, trial, config, ours):
    src, dst = trial['src'], trial['dst']
    expected_hash = input_hash(src, dst)
    # Both backends get numerically identical values and correspondence order.
    # Conversion, copies, and host/device transfer are outside the timer.
    if method == 'TEASER++':
        a, b = np.array(src.T, dtype=np.float64, order='F'), np.array(dst.T, dtype=np.float64, order='F')
    else:
        a, b = torch.from_numpy(src.copy()).to(config['device']), torch.from_numpy(dst.copy()).to(config['device'])
    record = dict(method=method, input_sha256=expected_hash, status='failed',
                  transformation=None, scale_error=None, rotation_error_deg=None, translation_error=None)
    p, o = backend_settings(config, ours)
    if config['device'].startswith('cuda') and method == 'Ours':
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        if method == 'TEASER++':
            solver = teaserpp_python.RobustRegistrationSolver(p)
            solver.solve(a, b)
        else:
            result = turboreg_gpu.register_sim3(a, b, o)
            if config['device'].startswith('cuda'):
                torch.cuda.synchronize()
        record['runtime_s'] = time.perf_counter() - start
        if method == 'TEASER++':
            result = solver.getSolution()
            if len(solver.getInlierMaxClique()) <= 1:
                raise ValueError('TEASER aborted: clique size <= 1 (valid flag not exposed in Python)')
            if hasattr(result, 'valid') and not result.valid:
                raise ValueError('TEASER reports invalid solution')
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = result.scale * result.rotation, result.translation
        else:
            T = result.transformation.cpu().numpy().astype(np.float64)
        if np.isfinite(T).all():
            record['transformation'] = T.tolist()
        record.update(errors(T, trial['gt']))
        record['status'] = 'ok'
    except Exception as exc:
        record.setdefault('runtime_s', time.perf_counter() - start)
        record['exception'] = f'{type(exc).__name__}: {exc}'
    if method == 'TEASER++':
        actual_hash = input_hash(a.T, b.T)
    else:
        actual_hash = input_hash(a.cpu().numpy(), b.cpu().numpy())
    if expected_hash != actual_hash or expected_hash != input_hash(src, dst):
        raise RuntimeError('Backend mutated shared inputs')
    return record


def environment():
    import teaserpp_python._teaserpp as native_teaser
    files = [Path(turboreg_gpu.__file__), Path(native_teaser.__file__)]
    cpu = next((x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines()
                if x.startswith('model name')), platform.processor())
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                torch=torch.__version__, teaser_version=importlib.metadata.version('teaserpp-python'),
                teaser_install_origin=importlib.metadata.distribution('teaserpp-python').read_text('direct_url.json'),
                cpu=cpu, platform=platform.platform(), cuda_available=torch.cuda.is_available(),
                affinity=sorted(os.sched_getaffinity(0)),
                native_sha256={str(p): digest(p) for p in files},
                teaser_omp_max_threads=getattr(teaserpp_python, 'OMP_MAX_THREADS', None),
                build_cache_sha256=digest(REPO / 'build/teaser_paper/CMakeCache.txt'),
                thread_environment={k: os.environ[k] for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_DYNAMIC']})


def load_models(root, config):
    import open3d as o3d
    manifest = json.loads((root / 'stanford/manifest.json').read_text())
    result = {}
    for obj in config['objects']:
        item = manifest['objects'][obj]
        path = root / item['selected_ply']
        if digest(path) != item['ply_sha256']:
            raise ValueError(f'PLY hash changed: {path}')
        vertices = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < config['n_points'] or not np.isfinite(vertices).all():
            raise ValueError(f'Invalid PLY: {path}')
        result[obj] = vertices
    return result, manifest


def saved_trial(path, generated):
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            trial = {k: f[k] for k in f.files}
        if set(trial) != set(generated) or any(not np.array_equal(trial[k], generated[k]) for k in trial):
            raise ValueError(f'Saved trial differs from deterministic regeneration: {path}')
        return trial
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **generated)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return generated


def read_records(path, repair=False):
    """Only truncate an incomplete final write; reject corruption and duplicate keys."""
    rows, seen = [], set()
    if not path.exists():
        return rows
    with open(path, 'rb+') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair:
                    f.truncate(pos)
                    break
                raise ValueError('Interrupted last record: resume run to repair')
            row = json.loads(line)
            key = (row['trial_id'], row['method'])
            if key in seen:
                raise ValueError(f'Duplicate result: {key}')
            seen.add(key)
            rows.append(row)
    return rows


@contextmanager
def exclusive(output):
    output.mkdir(parents=True, exist_ok=True)
    with open(output / '.lock', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate(config, root, output):
    models, _ = load_models(root, config)
    base = json.loads((REPO / 'configs/sim3.json').read_text())
    base.update(max_points=100, inlier_threshold=config['beta'])
    # Small predeclared grid; same 16 independent validation trials for every candidate.
    candidates = []
    for epsilon in [0.1, 0.2]:
        for neighbors in [20, 80]:
            candidates.append(dict(base, epsilon_scale=epsilon, k_neighbors=neighbors))
    with exclusive(output):
        trials = [make_trial(models['bunny'], config, 0, pct, r, 'validation')
                  for pct in config['outlier_percent'] for r in range(4)]
        (output / 'trials').mkdir(exist_ok=True)
        for i, trial in enumerate(trials):
            saved_trial(output / 'trials' / f'validation-{i:03d}.npz', trial)
        selection = []
        for idx, options in enumerate(candidates):
            rows = [solve('Ours', t, config, options) for t in trials]
            # Avoid introducing a success threshold: rank by mean geodesic rotation error,
            # assigning failed calls the maximal rotation error 180 degrees for selection only.
            score = float(np.mean([r['rotation_error_deg'] if r['status'] == 'ok' else 180 for r in rows]))
            selection.append(dict(candidate=idx, options=options, mean_rotation_deg_failure_180=score, results=rows))
            print(f'validation candidate {idx}: mean rotation {score:.6g} deg', flush=True)
        winner = min(selection, key=lambda x: (x['mean_rotation_deg_failure_180'], x['candidate']))
        # Exercise both adapters on independent noisy/outlier inputs before freezing.
        baseline = [solve('TEASER++', t, config, winner['options']) for t in trials]
        atomic_json(output / 'validation.json', dict(config=config, candidates=selection,
                    winner=winner['candidate'], teaser_smoke=baseline, environment=environment(),
                    trial_manifest=[dict(index=i, seed=t['seed'].tolist(), gt=t['gt'].tolist(),
                        input_sha256=input_hash(t['src'], t['dst'])) for i, t in enumerate(trials)]))
        atomic_json(output / 'ours_frozen.json', winner['options'])
        print(f"Frozen candidate {winner['candidate']} -> {output / 'ours_frozen.json'}", flush=True)


def run(config, root, output, options_file, limit=None):
    options = json.loads(options_file.read_text())
    models, manifest = load_models(root, config)
    env = environment()
    contract = dict(config=config, ours=options, dataset=manifest,
                    benchmark_sha256=digest(Path(__file__)), environment=env)
    contract_hash = hashlib.sha256(canonical(contract).encode()).hexdigest()
    with exclusive(output):
        (output / 'trials').mkdir(exist_ok=True)
        metadata = output / 'run.json'
        if metadata.exists():
            if json.loads(metadata.read_text())['contract_sha256'] != contract_hash:
                raise ValueError('Resume rejected: protocol, code, data, native binaries or environment changed')
        else:
            atomic_json(metadata, dict(contract_sha256=contract_hash, **contract))
            provenance = output / 'provenance'
            provenance.mkdir(exist_ok=True)
            shutil.copyfile(__file__, provenance / 'teaser_sim3_benchmark.py')
            shutil.copyfile(options_file, provenance / 'ours_frozen.json')
            shutil.copyfile(REPO / 'build/teaser_paper/CMakeCache.txt', provenance / 'teaser_CMakeCache.txt')
            shutil.copyfile(REPO / 'build/teaser_paper/compile_commands.json', provenance / 'teaser_compile_commands.json')
            for file in ['teaser/src/registration.cc', 'teaser/include/teaser/registration.h',
                         'test/benchmark/registration-benchmark.cc', 'LICENSE']:
                source = Path('/path/to/TEASER-plusplus') / file
                shutil.copyfile(source, provenance / ('teaser_' + source.name))
        rows = read_records(output / 'results.jsonl', repair=True)
        done = {(r['trial_id'], r['method']): r for r in rows}
        total = len(config['objects']) * len(config['outlier_percent']) * config['repeats'] * len(METHODS)
        print(f'Resume: {len(done)}/{total} method calls present', flush=True)
        # Untimed backend warm-up on independent seed; no test-GT input to either solver.
        warm = make_trial(models['bunny'], config, 0, 0, 10000, 'validation')
        for method in METHODS:
            solve(method, warm, config, options)
        calls = 0
        with open(output / 'results.jsonl', 'a') as log:
            for oi, obj in enumerate(config['objects']):
                for pct in config['outlier_percent']:
                    for rep in range(config['repeats']):
                        trial_id = f'{obj}-o{pct:02d}-r{rep:03d}'
                        generation_start = time.perf_counter()
                        generated = make_trial(models[obj], config, oi, pct, rep)
                        generation_s = time.perf_counter() - generation_start
                        io_start = time.perf_counter()
                        trial = saved_trial(output / 'trials' / f'{trial_id}.npz', generated)
                        io_s = time.perf_counter() - io_start
                        shared_hash = input_hash(trial['src'], trial['dst'])
                        # Alternate execution order to reduce systematic order/thermal effects.
                        for method in (METHODS if rep % 2 == 0 else METHODS[::-1]):
                            if (trial_id, method) in done:
                                if done[(trial_id, method)]['input_sha256'] != shared_hash:
                                    raise ValueError('Input hash differs from saved result')
                                continue
                            if limit is not None and calls >= limit:
                                print('Smoke call limit reached; formal run can resume.', flush=True)
                                return
                            row = solve(method, trial, config, options)
                            row.update(trial_id=trial_id, object=obj, outlier_percent=pct, repeat=rep,
                                       split='test', seed=trial['seed'].tolist(), gt=trial['gt'].tolist(),
                                       generation_s=generation_s, io_s=io_s, contract_sha256=contract_hash,
                                       scope='primary_fig6c' if obj == 'bunny' else 'extension_same_protocol')
                            log.write(canonical(row) + '\n')
                            log.flush()
                            os.fsync(log.fileno())
                            done[(trial_id, method)] = row
                            calls += 1
                            if len(done) % 40 == 0 or row['status'] != 'ok':
                                print(f'{len(done)}/{total} {trial_id} {method}: {row["status"]}', flush=True)
        print(f'Completed {len(done)}/{total} method calls', flush=True)


def summarize(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_records(output / 'results.jsonl')
    meta = json.loads((output / 'run.json').read_text())
    config = meta['config']
    expected = {(f'{o}-o{p:02d}-r{r:03d}', m)
                for o in config['objects'] for p in config['outlier_percent']
                for r in range(config['repeats']) for m in METHODS}
    actual = {(r['trial_id'], r['method']) for r in rows}
    if not actual <= expected:
        raise ValueError('Unexpected trials in results')
    for row in rows:
        if row['contract_sha256'] != meta['contract_sha256']:
            raise ValueError('Mixed result contracts')
    by_trial = {}
    for r in rows:
        by_trial.setdefault(r['trial_id'], set()).add(r['input_sha256'])
    if any(len(v) != 1 for v in by_trial.values()):
        raise ValueError('Methods did not use the same inputs')
    table = []
    for obj in config['objects']:
        fig, axes = plt.subplots(5, 1, figsize=(8, 13), constrained_layout=True)
        for mi, method in enumerate(METHODS):
            color = ['#2166ac', '#d6604d'][mi]
            for pi, pct in enumerate(config['outlier_percent']):
                group = [r for r in rows if (r['object'], r['method'], r['outlier_percent']) == (obj, method, pct)]
                failed = sum(r['status'] != 'ok' for r in group)
                item = dict(object=obj, scope='primary_fig6c' if obj == 'bunny' else 'extension_same_protocol',
                            outlier_percent=pct, method=method, expected=config['repeats'],
                            observed=len(group), failures=failed, missing=config['repeats'] - len(group))
                position = pi + (mi - .5) * .32
                for ki, key in enumerate(METRICS):
                    values = [r[key] for r in group if r[key] is not None]
                    if values:
                        stats = np.percentile(values, [25, 50, 75])
                        for suffix, value in zip(['q1_finite', 'median_finite', 'q3_finite'], stats):
                            item[f'{key}_{suffix}'] = float(value)
                        item[f'{key}_mean_all'] = ('incomplete' if len(group) != config['repeats'] else
                            '+inf' if failed and key != 'runtime_s' else float(np.mean(values)))
                        axes[ki].boxplot([values], positions=[position], widths=.25, whis=1.5,
                            patch_artist=True, showfliers=True,
                            boxprops=dict(facecolor=color, alpha=.45), medianprops=dict(color='black'),
                            flierprops=dict(marker='.', markersize=3, markerfacecolor=color, markeredgecolor=color))
                    else:
                        for suffix in ['q1_finite', 'median_finite', 'q3_finite']:
                            item[f'{key}_{suffix}'] = None
                        item[f'{key}_mean_all'] = '+inf' if group else 'incomplete'
                axes[4].bar(position, failed, width=.25, color=color, label=method if pi == 0 else None)
                table.append(item)
        for ax, ylabel in zip(axes, ['Absolute scale error', 'Rotation error [deg]',
                                    'Translation error [normalized units]', 'Time [s]', 'Invalid model / exception count']):
            ax.set_xticks(range(len(config['outlier_percent'])), config['outlier_percent'])
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=.25)
        for ax in axes[:4]:
            ax.set_yscale('log')
        axes[4].set_ylim(0, config['repeats'] + 1)
        axes[4].legend()
        axes[4].set_xlabel('Outlier rate [%]')
        fig.suptitle(f'{obj}: unknown scale, N=100, 40 trials per condition\n'
                     + ('Fig. 6(c) protocol with documented assumptions' if obj == 'bunny' else 'Extension: same Sim(3) protocol')
                     + f' | {len(rows)}/{len(expected)} total calls')
        fig.savefig(output / f'{obj}_boxplots.png', dpi=170)
        fig.savefig(output / f'{obj}_boxplots.pdf')
        plt.close(fig)
    table.sort(key=lambda x: (config['objects'].index(x['object']), x['outlier_percent'], METHODS.index(x['method'])))
    with open(output / 'summary.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    atomic_json(output / 'summary.json', dict(complete=actual == expected, expected_calls=len(expected),
                observed_calls=len(rows), shared_input_verified=True, groups=table))
    lines = ['# TEASER++ unknown-scale Sim(3) results', '',
             f'Calls: {len(rows)}/{len(expected)}; complete={actual == expected}.', '',
             'Paper Fig. 6(c) conditions, with documented sampling/mesh/plotting assumptions. '
             'Bunny is the primary reproduction; other objects are extensions. CPU, one thread. '
             'Not a bit-exact reproduction of unpublished author trial data.', '',
             'Each cell reports median [Q1, Q3] over finite estimates, including inaccurate estimates. '
             'Invalid/exception counts are explicit; no RR filter. CSV mean_all includes all trials '
             '(+inf for any invalid estimate; incomplete for missing trials). Translation uses the normalized coordinate unit '
             '(paper labels this m). Table is our summary of the boxplots, not a table published by TEASER.', '']
    for obj in config['objects']:
        lines += [f'## {obj}' + (' — primary Fig. 6(c)' if obj == 'bunny' else ' — extension'), '']
        for pct in config['outlier_percent']:
            lines += [f'### {pct}% outliers', '', '| Method | Scale error | Rotation (deg) | Translation | Time (s) | Invalid / observed |',
                      '|---|---:|---:|---:|---:|---:|']
            for method in METHODS:
                item = next(x for x in table if (x['object'], x['outlier_percent'], x['method']) == (obj, pct, method))
                cells = []
                for key in METRICS:
                    q1, med, q3 = [item[f'{key}_{s}'] for s in ['q1_finite', 'median_finite', 'q3_finite']]
                    cells.append('undefined' if med is None else f'{med:.5g} [{q1:.5g}, {q3:.5g}]')
                lines.append(f'| {method} | ' + ' | '.join(cells) + f' | {item["failures"]}/{item["observed"]} |')
            lines += ['']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(f'Summary: {output / "RESULTS.md"}, complete={actual == expected}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['validate', 'run', 'summarize'])
    p.add_argument('--config', type=Path, default=REPO / 'configs/teaser_sim3_protocol.json')
    p.add_argument('--data-root', type=Path, default=Path('/path/to/data/sim3_benchmarks'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ours-options', type=Path)
    p.add_argument('--limit-calls', type=int, help='Pipeline smoke only; does not change formal repeat count')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    configure_runtime(config)
    if args.command == 'validate':
        validate(config, args.data_root, args.output)
    elif args.command == 'run':
        if args.ours_options is None:
            p.error('run requires --ours-options from independent validation')
        run(config, args.data_root, args.output, args.ours_options, args.limit_calls)
    else:
        with exclusive(args.output):
            summarize(args.output)


if __name__ == '__main__':
    main()

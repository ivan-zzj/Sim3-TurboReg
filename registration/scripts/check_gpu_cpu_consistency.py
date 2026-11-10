#!/usr/bin/env python3
"""Join GPU vs CPU native results on identical frozen inputs and report agreement."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_cpu(path, method):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r['method'] != method:
                continue
            rows[r['input_sha256']] = r
    return rows

def load_gpu(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r['input_sha256']] = r
    return rows

def report(name, cpu_path, gpu_path, method='ours_native_cpu'):
    cpu = load_cpu(ROOT / cpu_path, method)
    gpu = load_gpu(ROOT / gpu_path)
    common = sorted(set(cpu) & set(gpu))
    only_cpu = sorted(set(cpu) - set(gpu))
    only_gpu = sorted(set(gpu) - set(cpu))
    print(f'=== {name} ===')
    print(f'cpu rows: {len(cpu)}, gpu rows: {len(gpu)}, common: {len(common)}, '
          f'only_cpu: {len(only_cpu)}, only_gpu: {len(only_gpu)}')
    cpu_fail = {h for h in common if cpu[h]['status'] != 'ok'}
    gpu_fail = {h for h in common if gpu[h]['status'] != 'ok'}
    print(f'cpu failures: {len(cpu_fail)}, gpu failures: {len(gpu_fail)}, '
          f'both fail: {len(cpu_fail & gpu_fail)}, cpu-only fail: {len(cpu_fail - gpu_fail)}, '
          f'gpu-only fail: {len(gpu_fail - cpu_fail)}')
    ok = [h for h in common if h not in cpu_fail and h not in gpu_fail]
    if not ok:
        return
    dre, dse, dte = [], [], []
    exact_T = 0
    for h in ok:
        c, g = cpu[h], gpu[h]
        dre.append(abs(c['rotation_error_deg'] - g['rotation_error_deg']))
        dse.append(abs(c['scale_error'] - g['scale_error']))
        dte.append(abs(c['translation_error'] - g['translation_error']))
        Tc = np.array(c['transformation']); Tg = np.array(g['transformation'])
        if np.allclose(Tc, Tg, atol=1e-6):
            exact_T += 1
    dre, dse, dte = map(np.array, (dre, dse, dte))
    print(f'both-ok inputs: {len(ok)}, identical transforms (atol 1e-6): {exact_T}')
    for label, arr in [('|dRE| deg', dre), ('|dScaleErr|', dse), ('|dTE|', dte)]:
        print(f'  {label}: median {np.median(arr):.3e}, p95 {np.quantile(arr, .95):.3e}, max {arr.max():.3e}')
    # largest disagreements
    idx = np.argsort(dre)[-5:][::-1]
    print('  largest |dRE| cases:')
    for i in idx:
        h = ok[i]
        c, g = cpu[h], gpu[h]
        print(f'    {c.get("input_path", h[:12])}: cpu RE={c["rotation_error_deg"]:.4f} '
              f'gpu RE={g["rotation_error_deg"]:.4f} cpuSE={c["scale_error"]:.5f} gpuSE={g["scale_error"]:.5f}')
    if gpu_fail:
        print('  gpu-failing inputs (first 10):')
        for h in sorted(gpu_fail)[:10]:
            g = gpu[h]
            c = cpu.get(h)
            cinfo = f'cpu status={c["status"]} RE={c.get("rotation_error_deg")}' if c else 'cpu row missing'
            print(f'    {g["input"]}: gpu error={g.get("error", "?")[:80]} | {cinfo}')

if __name__ == '__main__':
    report('comparison (1280)',
           'results/sim3_completion_final_20260916/comparison/results.jsonl',
           'results/ours_gpu_sim3_20260918/comparison/results.jsonl')
    report('sensitivity (1920)',
           'results/sim3_completion_final_20260916/sensitivity_complete/results.jsonl',
           'results/ours_gpu_sim3_20260918/sensitivity/results.jsonl')

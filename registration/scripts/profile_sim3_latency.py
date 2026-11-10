#!/usr/bin/env python3
"""Measure unprofiled latency and collect numeric CPU/CUDA traces, without plots."""
import argparse
import collections
import json
import os
from pathlib import Path
import time
import teaser_sim3_benchmark as b


def union_duration(intervals):
    total, end = 0.0, float('-inf')
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=b.REPO / 'results/teaser_tuning/profile')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--variant', choices=['original', 'tuned', 'both'], default='both')
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    config = json.loads((b.REPO / 'configs/teaser_sim3_protocol.json').read_text())
    b.configure_runtime(config)
    torch, np = b.torch, b.np
    torch.cuda.init()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = []
    for pct in [60, 70, 80, 90]:
        path = b.REPO / f'results/teaser_tuning/holdout/trials/bunny-o{pct}-r000.npz'
        with np.load(path) as trial:
            inputs.append((torch.from_numpy(trial['src'].copy()).cuda(), torch.from_numpy(trial['dst'].copy()).cuda()))
    summaries = []
    for label, filename in [('original', 'teaser_sim3_ours_frozen.json'), ('tuned', 'teaser_sim3_ours_tuned.json')]:
        if args.variant != 'both' and label != args.variant:
            continue
        options = json.loads((b.REPO / 'configs' / filename).read_text())
        _, native_options = b.backend_settings(config, options)
        for _ in range(20):
            b.turboreg_gpu.register_sim3(*inputs[0], native_options)
        torch.cuda.synchronize()
        latency = []
        for a, dst in inputs:
            samples = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()
                b.turboreg_gpu.register_sim3(a, dst, native_options)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            latency.append(float(np.median(samples)))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as prof:
            for index, (a, dst) in enumerate(inputs):
                torch.cuda.synchronize()
                with torch.profiler.record_function(f'registration_{index}'):
                    b.turboreg_gpu.register_sim3(a, dst, native_options)
                    torch.cuda.synchronize()
        trace_path = args.output / f'{label}_trace.json'
        prof.export_chrome_trace(str(trace_path))
        averages = prof.key_averages()
        (args.output / f'{label}_cpu.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=40))
        (args.output / f'{label}_cuda.txt').write_text(averages.table(sort_by='self_device_time_total', row_limit=40))
        ops = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
                    total_cpu_us=e.cpu_time_total, self_device_us=e.self_device_time_total,
                    total_device_us=e.device_time_total) for e in averages]
        b.atomic_json(args.output / f'{label}_operators.json', ops)
        events = json.loads(trace_path.read_text())['traceEvents']
        kernels = [e for e in events if e.get('cat') == 'kernel' and 'dur' in e]
        copies = [e for e in events if e.get('cat') == 'gpu_memcpy' and 'dur' in e]
        runtime = [e for e in events if e.get('cat') == 'cuda_runtime' and 'dur' in e]
        regions = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('registration_') and 'dur' in e]
        per_call = []
        for region in regions:
            inside = lambda es: [e for e in es if region['ts'] <= e['ts'] < region['ts'] + region['dur']]
            k, cp, rt = inside(kernels), inside(copies), inside(runtime)
            by_name = collections.Counter(e['name'] for e in rt)
            per_call.append(dict(name=region['name'], instrumented_wall_us=region['dur'],
                                 kernel_count=len(k), kernel_sum_us=sum(e['dur'] for e in k),
                                 kernel_union_us=union_duration([(e['ts'],e['ts']+e['dur']) for e in k]),
                                 memcpy_count=len(cp), memcpy_sum_us=sum(e['dur'] for e in cp),
                                 memcpy_directions=dict(collections.Counter(e['name'] for e in cp)),
                                 cuda_runtime_calls=dict(by_name),
                                 cuda_runtime_sum_us=sum(e['dur'] for e in rt)))
        summary = dict(label=label, unprofiled_median_ms_by_outliers=dict(zip([60,70,80,90],latency)),
                       profiled_calls=per_call,
                       key_ops={e['name']: e for e in ops if e['name'] in ['aten::item','aten::nonzero','aten::_local_scalar_dense',
                           'aten::copy_','aten::_linalg_svd','aten::linalg_svd','aten::topk','aten::bmm','aten::mm','aten::linalg_det']})
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    b.atomic_json(args.output / 'summary.json', dict(
        note='Profiler perturbs wall time; use unprofiled timing for latency. Kernel durations are from traces, not a subtraction-based latency decomposition.',
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, variants=summaries))


if __name__ == '__main__':
    main()

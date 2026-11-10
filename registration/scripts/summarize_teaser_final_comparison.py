#!/usr/bin/env python3
"""Audit the two-method replay and write final numeric tables, without plots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import teaser_sim3_benchmark as b
from tune_teaser_sim3 import stats

b.np = np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=b.REPO / 'results/teaser_sim3_final_comparison')
    args = parser.parse_args()
    out = args.output
    read = lambda p: json.loads(p.read_text())
    meta = read(out / 'run.json')
    assert read(out / 'summary.json')['complete']
    assert meta['split'] == 'retrospective'
    config = meta['config']
    assert config['repeats'] == 40 and config['n_points'] == 100
    frozen = read(b.REPO / 'results/teaser_tuning_final/frozen_selection.json')
    assert b.digest(b.REPO / 'configs/teaser_sim3_ours_final.json') == frozen['parameter_sha256']
    original = Path(meta['original_inputs'])
    assert b.digest(original / 'results.jsonl') == meta['original_results_sha256']
    ids = ['teaser_cpu', 'ours_balanced_gpu']
    assert [c['id'] for c in meta['candidates']] == ids
    results = [read(out / (cid + '.json')) for cid in ids]
    expected = {f'{obj}-o{pct:02d}-r{rep:03d}' for obj in config['objects']
                for pct in config['outlier_percent'] for rep in range(40)}
    reference = {r['trial_id']: r for r in results[0]['rows']}
    labels = ['TEASER++ CPU', 'Ours GPU（均衡参数）']
    metrics = ['scale_error', 'rotation_error_deg', 'translation_error']
    for result, candidate in zip(results, meta['candidates']):
        assert result['fingerprint'] == meta['fingerprint']
        assert result['candidate'] == candidate
        assert candidate['options'] == frozen['selected']['options']
        assert len(result['rows']) == len(expected) == 640
        assert {r['trial_id'] for r in result['rows']} == expected
        for row in result['rows']:
            tid = row['trial_id']
            ref = reference[tid]
            assert row['input_sha256'] == ref['input_sha256'] and row['gt'] == ref['gt']
            with np.load(out / 'trials' / (tid + '.npz')) as trial, np.load(original / 'trials' / (tid + '.npz')) as old:
                assert set(trial.files) == set(old.files)
                for key in trial.files:
                    assert np.array_equal(trial[key], old[key]), (tid, key)
                assert b.input_hash(trial['src'], trial['dst']) == row['input_sha256']
                assert np.array_equal(trial['gt'], np.asarray(row['gt']))
                assert trial['seed'].tolist() == row['seed']
            assert np.isfinite(row['runtime_s']) and row['runtime_s'] > 0
            if row['status'] == 'ok':
                checked = b.errors(np.asarray(row['transformation']), np.asarray(row['gt']))
                for key in metrics:
                    assert np.isclose(checked[key], row[key], atol=1e-10, rtol=1e-10)
            else:
                assert all(row[key] is None for key in metrics)
        checked = stats(result['rows'])
        for key, value in checked.items():
            assert result['summary'][key] == value
        for c in result['conditions']:
            rows = [r for r in result['rows'] if r['object'] == c['object'] and r['outlier_percent'] == c['outlier_percent']]
            assert len(rows) == 40
            for key, value in stats(rows).items():
                assert c[key] == value

    flat = []
    def summary(rows, scope, obj, pct, label):
        s = stats(rows)
        row = dict(scope=scope, object=obj, outlier_percent=pct, method=label,
                   trials=s['n'], failures=s['failures'])
        for key in metrics + ['runtime_s']:
            row[key + '_mean_all'] = s[key + '_mean_all']
            values = [r[key] for r in rows if r[key] is not None]
            for quantile, name in [(.25, 'q1_finite'), (.5, 'median_finite'), (.75, 'q3_finite')]:
                row[key + '_' + name] = float(np.quantile(values, quantile)) if values else None
        flat.append(row)
        return row

    def table(rows):
        lines = ['| 方法 | 平均绝对尺度误差 | 平均旋转误差° | 平均平移误差 | 时延中位数 ms | 求解异常 |',
                 '|---|---:|---:|---:|---:|---:|']
        fmt = lambda x: f'{x:.6g}' if isinstance(x, (float, int)) else str(x)
        for row in rows:
            values = [row[key + '_mean_all'] for key in metrics] + [row['runtime_s_median_finite'] * 1000]
            lines.append('| ' + row['method'] + ' | ' + ' | '.join(map(fmt, values)) + f' | {row["failures"]}/{row["trials"]} |')
        return lines

    lines = ['# TEASER++ 与均衡参数 Ours：最终重跑对比', '',
             '两种方法均已实际重新运行。输入来自最初保存的 640 份 NPZ；Ours 使用冻结的均衡配置，参数未再调整。', '',
             'Bunny 对应论文 Fig. 6(c) 的未知尺度主实验，Armadillo、Dragon、Buddha 为同协议扩展。'
             '原作者未公开的采样和 PLY 等细节仍使用已记录的复现假设，见 [协议](../../docs/TEASER_SIM3_PROTOCOL.md)。', '',
             'N=100，外点率 60/70/80/90%，每对象每外点率 40 次。σ=0.01、β=0.0554。'
             '报告全部试验的平均绝对尺度误差、SO(3) 测地旋转角（度）、平移欧氏误差；失败使全部试验误差均值为 +inf。'
             'CSV 另含有限估计的中位数与四分位数；所有有效但不准确的结果均保留，无联合 RR 门限。', '',
             'CPU 为 Intel Xeon Platinum 8481C；Ours 在 RTX 4090（GPU 1）运行。固定 1 线程；GPU 求解前后同步。'
             '时延包含求解 API 内部操作，排除生成、I/O、输入转换和初始 GPU 上传。', '',
             '**本次为已见输入上的回顾性重跑，不作为全新未见测试。**'
             '冻结参数后的两个独立种子验证见 [双种子报告](../../docs/TEASER_SIM3_FINAL_TUNING.md)。', '',
             '## Bunny 主实验概览（160 次，附加汇总）', '']
    primary = [summary([r for r in result['rows'] if r['object'] == 'bunny'], 'primary', 'bunny', 'all', label)
               for result, label in zip(results, labels)]
    lines += table(primary) + ['', '## 四对象概览（640 次，附加汇总）', '']
    overall = [summary(result['rows'], 'overall', 'all', 'all', label) for result, label in zip(results, labels)]
    lines += table(overall) + ['']
    for obj in config['objects']:
        lines += [f'## {obj}' + ('（主实验）' if obj == 'bunny' else '（同协议扩展）'), '']
        for pct in config['outlier_percent']:
            rows = [summary([r for r in result['rows'] if r['object'] == obj and r['outlier_percent'] == pct],
                            'condition', obj, pct, label) for result, label in zip(results, labels)]
            lines += [f'### 外点率 {pct}%（各 40 次）', ''] + table(rows) + ['']
    wins = []
    nonwins = []
    for a, ours in zip(flat[4::2], flat[5::2]):
        for key in metrics:
            left, right = a[key + '_mean_all'], ours[key + '_mean_all']
            target = wins if isinstance(right, (float, int)) and (left == '+inf' or right < left) else nonwins
            target.append(dict(object=a['object'], outlier_percent=a['outlier_percent'], metric=key, teaser=left, ours=right))
    max_re = max((r['rotation_error_deg'] for r in results[1]['rows'] if r['status'] == 'ok'), default=None)
    extra = dict(condition_metric_wins=len(wins), condition_metric_comparisons=len(wins)+len(nonwins),
                 nonwins=nonwins, ours_max_rotation_deg=max_re,
                 ours_to_teaser_median_latency_ratio=overall[1]['runtime_s_median_finite']/overall[0]['runtime_s_median_finite'])
    lines += ['## 可核验文件', '',
              '- [配置](../../configs/teaser_sim3_ours_final.json)',
              '- [逐条件均值、中位数、四分位数 CSV](comparison.csv)',
              '- [逐次结果 JSONL](results.jsonl)',
              '- [完整性审计](audit.json)', '',
              '复跑命令（项目根目录执行，GPU 需可见）：', '', '```bash',
              'python scripts/tune_teaser_sim3.py --candidates configs/teaser_sim3_final_comparison.json --output results/teaser_sim3_final_comparison --split retrospective --seed 2026091501 --repeats 40 --gpu 1 --original-inputs results/teaser_sim3_paper',
              'python scripts/summarize_teaser_final_comparison.py', '```', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    with (out / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with (out / 'results.jsonl').open('w') as handle:
        for pair in zip(*(result['rows'] for result in results)):
            for row in pair:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
    b.atomic_json(out / 'audit.json', dict(passed=True, trials_per_method=640, calls=1280,
                  original_npz_all_arrays_verified=True, shared_hash_gt_verified=True,
                  transform_errors_recomputed=True, condition_and_overall_statistics_recomputed=True,
                  frozen_parameter_sha256=frozen['parameter_sha256'], run_fingerprint=meta['fingerprint'],
                  report_script_sha256=b.digest(Path(__file__))))
    b.atomic_json(out / 'comparison_summary.json', dict(primary=primary, overall=overall, diagnostics=extra))
    print(json.dumps(dict(primary=primary, overall=overall, diagnostics=extra), indent=2))


if __name__ == '__main__':
    main()

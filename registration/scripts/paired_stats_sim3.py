#!/usr/bin/env python3
"""Paired statistics for the frozen Sim(3) comparison suite.

Per condition (object x outlier x seed), Ours is compared against each baseline
on identical inputs. Reports per-cell sign tests on rotation error and an
overall summary across the 96 cells, plus Wilcoxon signed-rank on per-cell mean
errors, for both rotation and scale error.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/sim3_completion_final_20260916/comparison/results.jsonl'
OUT = ROOT / 'results/sim3_completion_final_20260916/paired_stats.json'

rows = defaultdict(dict)  # input_sha256 -> method -> row
with open(RESULTS) as f:
    for line in f:
        r = json.loads(line)
        rows[r['input_sha256']][r['method']] = r

cells = defaultdict(lambda: defaultdict(list))  # (object,outlier,seed) -> method -> [row]
for h, m in rows.items():
    any_row = next(iter(m.values()))
    key = (any_row['object'], any_row['outlier_percent'], any_row['seed'])
    for meth, r in m.items():
        cells[key][meth].append(r)

def cell_pairs(cell, a, b, metric):
    xa, xb = [], []
    ra = {r['input_sha256']: r for r in cell.get(a, [])}
    rb = {r['input_sha256']: r for r in cell.get(b, [])}
    for h in sorted(set(ra) & set(rb)):
        if ra[h]['status'] == 'ok' and rb[h]['status'] == 'ok':
            xa.append(ra[h][metric]); xb.append(rb[h][metric])
    return np.array(xa), np.array(xb)

report = {}
for baseline in ['teaser_cpu', 'ransac_1min', 'fgr_sim3']:
    cell_summaries = []
    for key in sorted(cells):
        cell = cells[key]
        rec = {'object': key[0], 'outlier': key[1], 'seed': key[2]}
        for metric in ['rotation_error_deg', 'scale_error']:
            xa, xb = cell_pairs(cell, 'ours_native_cpu', baseline, metric)
            if len(xa) == 0:
                rec[metric] = None
                continue
            diff = xa - xb
            wins = int(np.sum(diff < 0)); losses = int(np.sum(diff > 0)); ties = int(np.sum(diff == 0))
            p_sign = stats.binomtest(min(wins, losses), wins + losses, 0.5).pvalue if wins + losses else 1.0
            rec[metric] = {
                'n_paired': len(xa), 'ours_mean': float(np.mean(xa)), 'base_mean': float(np.mean(xb)),
                'ours_wins': wins, 'losses': losses, 'ties': ties, 'sign_p': float(p_sign),
                'median_paired_diff': float(np.median(diff)),
            }
        cell_summaries.append(rec)
    # overall: Wilcoxon across per-cell mean differences
    overall = {}
    for metric in ['rotation_error_deg', 'scale_error']:
        d = np.array([c[metric]['ours_mean'] - c[metric]['base_mean']
                      for c in cell_summaries if c[metric]])
        cells_lower = int(np.sum(d < 0))
        w = stats.wilcoxon(d) if np.all(d != 0) else None
        overall[metric] = {
            'cells': len(d), 'cells_ours_lower': cells_lower,
            'sign_test_p': float(stats.binomtest(min(cells_lower, len(d) - cells_lower), len(d), 0.5).pvalue),
            'wilcoxon_p': float(w.pvalue) if w else None,
            'median_cell_diff': float(np.median(d)),
        }
    report[baseline] = {'per_cell': cell_summaries, 'overall': overall}

OUT.write_text(json.dumps(report, indent=2) + '\n')
for baseline, rep in report.items():
    print(f'=== Ours vs {baseline} ===')
    for metric, o in rep['overall'].items():
        print(f"  {metric}: {o['cells_ours_lower']}/{o['cells']} cells lower, "
              f"sign p={o['sign_test_p']:.3e}, wilcoxon p={o['wilcoxon_p']:.3e}, "
              f"median cell diff={o['median_cell_diff']:.5f}")
print('wrote', OUT)

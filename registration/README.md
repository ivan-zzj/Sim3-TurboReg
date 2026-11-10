# Registration Experiments

Controlled point cloud registration experiments for the paper: the **TEASER++ unknown-scale Sim(3) protocol** (Bunny main experiment, Fig. 6(c) style; Armadillo / Dragon / Buddha as same-protocol extensions).

The full protocol, fixed reproduction assumptions, integrity checks, and timing boundaries are documented in [`../docs/TEASER_SIM3_PROTOCOL.md`](../docs/TEASER_SIM3_PROTOCOL.md). Please read it before re-running anything — several details not published by the TEASER++ authors are explicitly fixed there.

## Layout

- `configs/` — frozen experiment configurations (`teaser_sim3_ours_final.json` is the balanced parameter set used for the final comparison; `teaser_sim3_protocol.json` pins the protocol itself).
- `scripts/` — benchmark, evaluation, and analysis entry points:
  - `build_teaser_paper.sh` — build the pinned TEASER++ release for the benchmark
  - `prepare_teaser_stanford.py` — verify and unpack the Stanford 3D scanning models (CRC / path-safety checks)
  - `teaser_sim3_benchmark.py` — `validate` / `run` / `summarize` the Monte-Carlo benchmark (resumable, fsync'd JSONL)
  - `evaluate_sim3.py` — manifest-driven evaluation (`--method turboreg|ransac|teaser|se3|external`)
  - `benchmark_ours_gpu.py`, `benchmark_sim3_native.py` — GPU / CPU-native latency benchmarks
  - `generate_scale_variant_dataset.py` — synthetic scale-variant correspondence generator
  - `paired_stats_sim3.py`, `summarize_teaser_final_comparison.py`, `replay_teaser_sim3_gpu.py`, `check_gpu_cpu_consistency.py`, `profile_sim3_latency.py` — statistics, audit, and replay tooling
  - `build_sim3_comparison.py`, `build_sim3_native.py` — build helpers for the comparison / CPU-native variants

## Reproduce

```bash
# from the repository root, with bindings built and PYTHONPATH set
bash registration/scripts/build_teaser_paper.sh
python -m unittest discover -s tests -p test_teaser_protocol.py -v
python registration/scripts/teaser_sim3_benchmark.py validate --output results/teaser_sim3_validation
python registration/scripts/teaser_sim3_benchmark.py run \
  --output results/teaser_sim3_paper \
  --ours-options results/teaser_sim3_validation/ours_frozen.json
python registration/scripts/teaser_sim3_benchmark.py summarize --output results/teaser_sim3_paper
```

Re-running the same `run` command resumes from completed trials. Use a fresh output directory when the configuration changes.

## Headline results (final frozen run, 640 solver calls per method)

Unknown-scale Sim(3), N = 100 correspondences, outlier ratios 60–90%. Errors are arithmetic means over **all** trials; latency is the median synchronized solve call (TEASER++ on CPU, ours on an RTX 4090).

Bunny main experiment:

| Outlier ratio | Method | Scale err | Rotation err (deg) | Translation err | Latency (ms) |
|---:|---|---:|---:|---:|---:|
| 60% | TEASER++ | 0.0044 | 0.204 | 0.0073 | 0.89 |
| 60% | **Ours** | **0.0029** | **0.169** | **0.0060** | 3.01 |
| 70% | TEASER++ | 0.0096 | 0.237 | 0.0105 | 0.86 |
| 70% | **Ours** | **0.0036** | **0.174** | **0.0064** | 3.01 |
| 80% | TEASER++ | 0.0206 | 0.310 | 0.0184 | 0.85 |
| 80% | **Ours** | **0.0041** | **0.235** | **0.0085** | 3.01 |
| 90% | TEASER++ | 5.9169 | 120.72 | 6.8008 | 0.90 |
| 90% | **Ours** | **0.0075** | **0.483** | **0.0177** | 3.02 |

Across all 16 object × outlier conditions (4 objects), ours has lower mean error on 48/48 condition–metric pairs, with a worst-case rotation error of 1.92°. The trade-off: our per-call latency is ~3.5× TEASER++ on this small-N regime. See the paper for the full tables, the CPU-native variant, and failure analysis.

Raw run archives (JSONL / NPZ / audit logs, ~1 GB) are **not** shipped in this repository; every number above can be regenerated with the commands and frozen configs here.

# Streaming 3D Reconstruction

This directory contains our integration code that replaces the **IRLS chunk-alignment backend** of streaming 3D reconstruction pipelines with **Sim(3)-TurboReg**, as evaluated in the paper on KITTI Odometry and TUM RGB-D.

The upstream pipelines are **not** redistributed here — they carry their own licenses (VGGT-Long: custom VGGT license; Depth Anything 3: Apache-2.0; Pi3: its own license). Clone them separately and copy/apply our files on top.

## Layout

- `vggt_long/` — integration for **VGGT-Long** and **Pi-Long** (Pi-Long reuses the VGGT-Long driver with the `pi3_*` configs):
  - `vggt_long_turboreg.py` — drop-in pipeline variant replacing IRLS alignment with Sim(3)-TurboReg (confidence-filtered correspondences → robust Sim(3))
  - `eval_compare.py`, `eval_alignment_only.py` — ATE evaluation (global Sim(3) alignment, RMSE)
  - `configs/` — `turboreg_config.yaml` (gate on), `turboreg_nogate_config.yaml` (gate off), and the `pi3_*` equivalents
  - `run_compare.sh`, `run_compare_pi3.sh`, `run_batch.sh`, `run_batch_tum.sh` — per-sequence and batch runners
  - `scripts/compute_interchunk_scale_stats.py` — inter-chunk scale statistics used in the analysis
  - `patches/upstream_modifications.patch` — our changes to upstream files (`vggt_long.py`, `configs/base_config.yaml`), apply with `git apply` inside the VGGT-Long clone
- `da3_streaming/` — integration for **DA3-Streaming** (`da3_streaming/` inside the Depth-Anything-3 repo):
  - `da3_streaming_turboreg.py` — TurboReg backend variant
  - `eval_compare.py` — shared evaluation entry
  - `configs/` — default TurboReg configs (`turboreg_config.yaml`, `tum_turboreg.yaml`, overlap variants) and `configs/ablation/` (k / τ / ε / chunk-size ablations from the paper)
  - `run_*.sh`, `compare*.sh`, `stat_alignment_time*.sh` — batch, comparison, ablation, and timing scripts
  - `gen_ablation_config.py` — ablation config generator
  - `patches/upstream_modifications.patch` — our changes to upstream `da3_streaming.py` and configs

## Setup

1. Build the TurboReg Python bindings (repository root → `bindings/`), then either:
   - set `export TURBOREG_BINDINGS=/path/to/this/repo/bindings`, or
   - keep the default: the scripts look for `../../bindings` relative to their own location.
2. Clone the upstream pipelines:
   - `git clone https://github.com/DengKaiCQ/VGGT-Long.git`
   - `git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git`
   - Pi3 weights/configs per the VGGT-Long instructions.
3. Copy the files from `vggt_long/` into the VGGT-Long clone root, and the files from `da3_streaming/` into `Depth-Anything-3/da3_streaming/`; apply `patches/upstream_modifications.patch` with `git apply` in each clone.
4. Edit the `/path/to/...` placeholders in the YAML configs and shell scripts: model weights (`weights/`), datasets (`kitti/`, `tum/`), and output roots (`exps/`, `eval_results/`).

## Notes on configuration

- Gate policy used in the paper: VGGT-Long / Pi-Long — gate **on** for KITTI, **off** for TUM; DA3-Streaming — gate **on** for both.
- Paper defaults: KITTI — τ = 0.25, k = 10, chunk 120/60; TUM — τ = 0.20, k = 20, chunk 120/60. Ablation configs live in `da3_streaming/configs/ablation/`.
- Timing numbers in the paper are single-run wall-clock on one machine and are meant for coarse comparison only.

## Headline results

KITTI Odometry ATE RMSE (m), Avg / Avg\* excluding Seq 01; TUM RGB-D ATE RMSE (m):

| Method | KITTI Avg | KITTI Avg\* | TUM Avg |
|---|---:|---:|---:|
| VGGT-Long | 25.671 | 22.432 | 0.0824 |
| Pi-Long | 29.766 | 15.278 | 0.0808 |
| DA3-Streaming | 24.477 | 16.541 | 0.0492 |
| VGGT-Long + TurboReg | 29.303 | 25.026 | 0.0698 |
| Pi-Long + TurboReg | 28.141 | 15.428 | 0.0682 |
| DA3-Streaming + TurboReg | **19.046** | **13.756** | **0.0437** |

Swapping DA3-Streaming's backend to Sim(3)-TurboReg improves KITTI ATE by ~22% (24.477 → 19.046) and gives the best ATE on 7 of 9 evaluated sequences. VGGT-Long degrades on KITTI highway sequences without loop closures (Seq 01/03/08) — analyzed as a failure mode in the paper.

Raw per-run archives are not shipped; every table entry can be regenerated with the configs and scripts in this directory.

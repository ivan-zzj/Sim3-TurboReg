# Sim(3)-TurboReg

Official code for **"Sim(3)-TurboReg: Robust and Efficient Scale-Aware Registration via Tri-Scale Consistency"** (under review at IEEE RA-L; arXiv link coming soon).

Sim(3)-TurboReg registers outlier-contaminated 3D correspondences `src, dst ∈ R^(N×3)` under an unknown scale:

```
y = sRx + t,   s > 0,  R ∈ SO(3)
```

The core method combines local scale voting, a Tri-Scale Consistency (TSC) graph, and TurboClique candidate search. Graph construction, model solving, scoring, refinement, and the RANSAC baseline all run in C++/LibTorch (CPU and CUDA share the same dense-graph algorithm); Python handles data, scheduling, and evaluation. The solver takes **correspondences** as input — feature extraction and matching are provided by the caller.

The repository has two main parts:

- **`registration/`** — controlled point cloud registration experiments (TEASER++ unknown-scale Sim(3) protocol, Stanford 3D scanning benchmark, RANSAC baseline, GPU/CPU evaluation).
- **`streaming/`** — streaming 3D reconstruction: integrating Sim(3)-TurboReg as the chunk-alignment backend of VGGT-Long, Pi-Long, and DA3-Streaming (KITTI Odometry / TUM RGB-D).

```
├── turboreg/         core C++ library (headers in turboreg/include, sources in turboreg/src)
├── apps/             C++ CLI
├── bindings/         Python bindings (pybind11 / LibTorch)
├── registration/     point cloud registration experiments (configs, scripts)
├── streaming/        streaming 3D reconstruction integration (VGGT-Long / Pi-Long / DA3-Streaming)
├── tests/            C++ and Python tests
├── demo_data/        sample correspondence pairs
└── docs/             pipeline, experiment protocol, reproduction notes
```

## Build

Requirements: C++17, CMake ≥ 3.18, Eigen3, PyTorch/LibTorch. CUDA is optional; when configuring against a CUDA build of LibTorch you still need a matching CUDA toolkit.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Python bindings (same C++ sources, built separately):

```bash
cd bindings
MAX_JOBS=2 python setup.py build_ext --inplace
cd ..
export PYTHONPATH="$PWD/bindings${PYTHONPATH:+:$PYTHONPATH}"
```

or, inside an environment that already has PyTorch:

```bash
python -m pip install --no-build-isolation ./bindings
```

## Quick start (Python)

```python
import torch
import turboreg_gpu

options = turboreg_gpu.Sim3Options()
options.epsilon_scale = 0.1
options.inlier_threshold = 0.1   # in target-frame units; 0.1 = 10 cm if units are meters
options.max_points = 7000

src = torch.randn(100, 3)        # replace with real correspondences; can live on cuda:0
dst = 1.5 * src + torch.tensor([0.2, -0.1, 0.3])
result = turboreg_gpu.register_sim3(src, dst, options)
print(result.transformation, result.scale, result.num_inliers)

# RANSAC baseline, also running in C++:
baseline = turboreg_gpu.ransac_sim3(src, dst, options, iterations=10000, seed=42)
```

Inputs are finite `N×3` float tensors on the same device, `N ≥ 3`, with pairwise row correspondence. The solver raises on invalid input or when no valid hypothesis exists — it never silently falls back to identity or an all-points least-squares fit.

## C++ API and CLI

```cpp
#include <turboreg/sim3.hpp>

turboreg::Sim3Options options;
options.epsilon_scale = 0.1f;
auto result = turboreg::register_sim3(src, dst, options);
auto T = result.transformation;   // [sR, t; 0 1]
```

CMake subprojects can simply link the `turboreg` target. The CLI reads six-column `src_xyz dst_xyz` rows and writes a plain `4×4` matrix:

```bash
./build/turboreg_sim3 correspondences.txt transform.txt cpu
```

## Registration experiments

See [`registration/`](registration/README.md) for the TEASER++ unknown-scale Sim(3) protocol, dataset preparation, benchmark scripts, and evaluation entry points. The full protocol description is in [`docs/TEASER_SIM3_PROTOCOL.md`](docs/TEASER_SIM3_PROTOCOL.md).

## Streaming 3D reconstruction

See [`streaming/`](streaming/README.md). This part contains our integration code that replaces the IRLS chunk-alignment backend of [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long), Pi-Long, and [DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3) (`da3_streaming`) with Sim(3)-TurboReg, plus the configs and scripts for the KITTI Odometry / TUM RGB-D evaluation. The upstream pipelines must be cloned separately and are **not** redistributed here (they carry their own licenses).

## Authors

Zijun Zhou, Shaocheng Yan, Zhenjun Zhao, Yutong Hu, Ke Yang, Yingdong Gu, Pengcheng Shi, Jiayuan Li

- Wuhan University, China (all authors except Zhenjun Zhao)
- University of Zaragoza, Spain (Zhenjun Zhao)

## License

This repository is released under the [MIT License](LICENSE). Note that the upstream streaming pipelines (VGGT-Long, Depth Anything 3, Pi3) are governed by their own licenses — please check them before use.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sim3turboreg2026,
  title   = {Sim(3)-TurboReg: Robust and Efficient Scale-Aware Registration via Tri-Scale Consistency},
  author  = {Zhou, Zijun and Yan, Shaocheng and Zhao, Zhenjun and Hu, Yutong and Yang, Ke and Gu, Yingdong and Shi, Pengcheng and Li, Jiayuan},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

This project builds on the original TurboReg:

```bibtex
@article{yan2025turboreg,
  title   = {TurboReg: TurboClique for Robust and Efficient Point Cloud Registration},
  author  = {Yan, Shaocheng and Shi, Pengcheng and Zhao, Zhenjun and Wang, Kaixin and Cao, Kuang and Wu, Ji and Li, Jiayuan},
  journal = {arXiv preprint arXiv:2507.01439},
  year    = {2025}
}
```

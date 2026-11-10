#!/usr/bin/env bash
# Build the unmodified official backend in isolation; no environment installation.
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/.." && pwd)
teaser_source=${TEASER_SOURCE:-/home/zhouzijun/TEASER-plusplus}
benchmark_python=${BENCHMARK_PYTHON:-/mnt/data1/zhouzijun/envs/turboreg/bin/python}
cached_deps="$teaser_source/build/cp311-cp311-linux_x86_64/_deps"
expected_commit=52a9c52ee7d4c838c5e8a75458c33178be5bfb70
test "$(git -C "$teaser_source" rev-parse HEAD)" = "$expected_commit"
git -C "$teaser_source" diff --quiet HEAD -- teaser python CMakeLists.txt
if [ ! -f "$repo_root/build/teaser_build_deps/pybind11/share/cmake/pybind11/pybind11Config.cmake" ]; then
  "$benchmark_python" -m pip install --no-deps --target "$repo_root/build/teaser_build_deps" pybind11==2.13.6
fi
cmake -S "$teaser_source" -B "$repo_root/build/teaser_paper" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_TESTING=OFF -DBUILD_DOC=OFF -DENABLE_DIAGNOSTIC_PRINT=OFF \
  -Dpybind11_DIR="$repo_root/build/teaser_build_deps/pybind11/share/cmake/pybind11" \
  -DPYTHON_EXECUTABLE="$benchmark_python" \
  -DFETCHCONTENT_SOURCE_DIR_PMC="$cached_deps/pmc-src" \
  -DFETCHCONTENT_SOURCE_DIR_SPECTRA="$cached_deps/spectra-src" \
  -DFETCHCONTENT_SOURCE_DIR_TINYPLY="$cached_deps/tinyply-src"
cmake --build "$repo_root/build/teaser_paper" --target teaserpp_python -j2
cp "$teaser_source/python/teaserpp_python/__init__.py" "$repo_root/build/teaser_paper/python/teaserpp_python/__init__.py"

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <string>

namespace turboreg {

// Coordinates obey dst = scale * R * src + t. Distances use input units.
struct Sim3Options {
    int max_points = 7000;
    float epsilon_scale = 0.1f;
    int num_pivots = 2000;
    float nms_radius = 0.15f;
    float inlier_threshold = 0.1f;
    std::string metric = "IN";
    int k_neighbors = 20;
    int graph_iterations = 2;
    int refinement_iterations = 20;
    float min_pair_distance = 1e-3f;
    float scale_min = 0.1f;
    float scale_max = 10.0f;
    float log_scale_bin_width = 0.1f;
    std::string local_scale_mode = "hist";
    bool use_tsc = true;
    bool use_second_order = true;
    bool reject_degenerate = true;
};

struct Sim3Result {
    torch::Tensor transformation; // (4,4), upper block = sR
    torch::Tensor inlier_mask;    // (input N,), on input device
    int64_t num_input = 0;
    int64_t num_used = 0;         // first min(N, max_points) correspondences
    int64_t num_inliers = 0;
    double scale = 1.0;
};

// Stateless, no input mutation. Matching finite floating (N>=3,3) tensors.
// Runs in float32 on the input device. Invalid input / no hypothesis throws.
Sim3Result register_sim3(const torch::Tensor& src, const torch::Tensor& dst,
                         const Sim3Options& options = {});

// Fixed-budget, seeded 3-point RANSAC + Umeyama, implemented in C++.
Sim3Result ransac_sim3(const torch::Tensor& src, const torch::Tensor& dst,
                       const Sim3Options& options = {}, int iterations = 10000,
                       uint64_t seed = 42);

} // namespace turboreg

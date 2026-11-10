#pragma once
#include <Eigen/Core>
#include <string>
#include <vector>

namespace turboreg::native {
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
struct Options {
    int max_points=100, num_pivots=32, k_neighbors=5;
    int graph_iterations=1, refinement_iterations=1;
    double epsilon_scale=.1, nms_radius=.15, inlier_threshold=.08;
    double min_pair_distance=.001, scale_min=.1, scale_max=10, log_scale_bin_width=.1;
    std::string metric="MAE", local_scale_mode="hist";
    bool use_tsc=true, use_second_order=false, reject_degenerate=false;
    // Optional estimator change, explicitly selected and reported independently.
    bool uniform_refinement=false;
};
struct Result {
    Eigen::Matrix4d transformation;
    std::vector<unsigned char> inlier_mask;
    int num_input=0, num_used=0, num_inliers=0;
    double scale=1;
};
Result register_sim3(const Eigen::Ref<const Points>& src,
                     const Eigen::Ref<const Points>& dst, const Options& options={});
}

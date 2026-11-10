#include <turboreg/sim3.hpp>
#include <turboreg/scale_turboreg.hpp>
#include <algorithm>
#include <random>
#include <limits>

namespace turboreg {
namespace {
ScaleTurboRegGPU solver(const Sim3Options& o) {
    return ScaleTurboRegGPU(o.max_points, o.epsilon_scale, o.num_pivots,
        o.nms_radius, o.inlier_threshold, o.metric, o.k_neighbors,
        o.graph_iterations, 0.1f, o.min_pair_distance, o.scale_min, o.scale_max,
        o.log_scale_bin_width, o.local_scale_mode, o.use_tsc, false, -1.0f,
        o.use_second_order, o.reject_degenerate, o.refinement_iterations);
}

Sim3Result result(const torch::Tensor& T, const torch::Tensor& src,
                  const torch::Tensor& dst, const Sim3Options& o) {
    Sim3Result r;
    r.transformation = T;
    auto predicted = src.to(T.options()).matmul(T.slice(0,0,3).slice(1,0,3).t())
                   + T.index({torch::indexing::Slice(0,3), 3});
    r.inlier_mask = (predicted - dst.to(T.options())).norm(2,1) < o.inlier_threshold;
    r.num_input = src.size(0);
    r.num_used = std::min<int64_t>(r.num_input, o.max_points);
    // One small device-to-host transfer for all public scalar fields and checks.
    auto stats = torch::stack({torch::isfinite(T).all().to(torch::kFloat64),
        T.index({torch::indexing::Slice(0,3), 0}).norm().to(torch::kFloat64),
        r.inlier_mask.sum().to(torch::kFloat64)}).to(torch::kCPU);
    const auto* values = stats.data_ptr<double>();
    TORCH_CHECK(values[0] != 0, "Non-finite Sim3 estimate.");
    r.scale = values[1];
    TORCH_CHECK(r.scale > 0, "Invalid Sim3 scale.");
    r.num_inliers = static_cast<int64_t>(values[2]);
    TORCH_CHECK(r.num_inliers >= 3, "No Sim3 consensus with at least three correspondences.");
    return r;
}

torch::Tensor triangle_mask(const torch::Tensor& points) {
    auto a = points.select(1,1) - points.select(1,0);
    auto b = points.select(1,2) - points.select(1,0);
    auto c = points.select(1,2) - points.select(1,1);
    auto max_edge2 = std::get<0>(torch::stack({a.square().sum(1), b.square().sum(1), c.square().sum(1)},1).max(1));
    return torch::cross(a,b,1).norm(2,1) > 1e-4f * max_edge2;
}
} // namespace

Sim3Result register_sim3(const torch::Tensor& src, const torch::Tensor& dst, const Sim3Options& o) {
    torch::NoGradGuard guard;
    auto estimator = solver(o);
    auto a = src, b = dst; // legacy normalization only rebinds these handles
    auto T = estimator.runRegCXXReturnTensor(a,b);
    return result(T,src,dst,o);
}

Sim3Result ransac_sim3(const torch::Tensor& src, const torch::Tensor& dst,
                       const Sim3Options& o, int iterations, uint64_t seed) {
    torch::NoGradGuard guard;
    auto validated_options = solver(o);
    (void)validated_options;
    TORCH_CHECK(iterations > 0, "RANSAC iterations must be positive.");
    TORCH_CHECK(src.dim()==2 && src.sizes()==dst.sizes() && src.size(1)==3 && src.size(0)>=3,
                "RANSAC expects matching (N>=3,3) tensors.");
    TORCH_CHECK(src.device()==dst.device() && src.is_floating_point() && dst.is_floating_point(),
                "Inputs must be floating tensors on the same device.");
    TORCH_CHECK(torch::isfinite(src).all().item<bool>() && torch::isfinite(dst).all().item<bool>(),
                "Inputs must be finite.");
    int64_t n = std::min<int64_t>(src.size(0),o.max_points);
    auto a=src.slice(0,0,n).to(torch::kFloat32).contiguous();
    auto b=dst.slice(0,0,n).to(torch::kFloat32).contiguous();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> index(0,n-1);
    torch::Tensor best;
    double best_score=-1;
    ModelSelection selection(o.metric,o.inlier_threshold);
    for (int start=0;start<iterations;start+=128) {
        int count=std::min(128,iterations-start);
        std::vector<int64_t> ids(count*3);
        for(int i=0;i<count;++i) {
            auto& x=ids[3*i]; auto& y=ids[3*i+1]; auto& z=ids[3*i+2];
            x=index(rng); do { y=index(rng); } while(y==x);
            do { z=index(rng); } while(z==x || z==y);
        }
        auto idx=torch::from_blob(ids.data(),{count*3},torch::kInt64).clone().to(a.device());
        auto sa=a.index_select(0,idx).view({count,3,3});
        auto sb=b.index_select(0,idx).view({count,3,3});
        auto valid=triangle_mask(sa)&triangle_mask(sb);
        if (!valid.any().item<bool>()) continue;
        auto models=umeyama_transform_3d(sa.index({valid}),sb.index({valid}));
        auto chosen=selection.calculate_best_clique(models,a,b).item<int64_t>();
        auto T=models[chosen];
        auto residual=(a.matmul(T.slice(0,0,3).slice(1,0,3).t())+T.index({torch::indexing::Slice(0,3),3})-b).norm(2,1);
        double score;
        if(o.metric=="IN") score=(residual<o.inlier_threshold).sum().item<double>();
        else {
            auto w=(1-residual/o.inlier_threshold).clamp_min(0);
            score=(o.metric=="MSE"?w.square():w).sum().item<double>();
        }
        if(score>best_score) { best_score=score; best=T.clone(); }
    }
    TORCH_CHECK(best.defined(), "RANSAC found no nondegenerate hypothesis.");
    best=post_refinement_sim3(best,a,b,o.refinement_iterations,o.inlier_threshold);
    return result(best,src,dst,o);
}
} // namespace turboreg

#include <turboreg/scale_turboreg.hpp>
#include <turboreg/detail/registration_utils.hpp>
#include <turboreg/utils_debug.hpp>
#ifdef TURBOREG_WITH_CUDA_KERNELS
#include <turboreg/detail/sim3_cuda.hpp>
#endif
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <tuple>
#include <limits>

namespace turboreg
{

    static ScaleTurboRegGPU::LocalScaleMode parse_local_scale_mode(const std::string &mode)
    {
        std::string m;
        m.reserve(mode.size());
        for (char c : mode)
        {
            m.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (m == "median" || m == "med" || m == "0")
            return ScaleTurboRegGPU::LocalScaleMode::Median;
        if (m == "union_median" || m == "median_union" || m == "union-med" || m == "unionmed" || m == "2")
            return ScaleTurboRegGPU::LocalScaleMode::UnionMedian;
        if (m == "hist" || m == "histogram" || m == "1")
            return ScaleTurboRegGPU::LocalScaleMode::Histogram;
        throw std::invalid_argument("local_scale_mode must be hist, median or union_median");
    }

    ScaleTurboRegGPU::ScaleTurboRegGPU(int max_N, float epsilon_scale, int num_pivot,
                                       float radiu_nms, float tau_inlier,
                                       const std::string &metric_str,
                                       int k_neighbors, int num_iterations,
                                       float trim_ratio, float min_pair_dist,
                                       float local_scale_min, float local_scale_max,
                                       float local_scale_diff,
                                       const std::string &local_scale_mode,
                                       bool use_tsc,
                                       bool use_distance_diff_consistency,
                                       float distance_diff_thresh,
                                       bool use_second_scoring,
                                       bool use_degeneracy_filter,
                                       int refinement_iterations)
        : max_N(max_N), epsilon_scale(epsilon_scale), num_pivot(num_pivot),
          radiu_nms(radiu_nms), tau_inlier(tau_inlier),
          k_neighbors(k_neighbors), num_iterations(num_iterations),
          trim_ratio(trim_ratio), min_pair_dist(min_pair_dist),
          local_scale_min(local_scale_min), local_scale_max(local_scale_max),
          local_scale_diff(local_scale_diff),
          local_scale_mode(parse_local_scale_mode(local_scale_mode)),
          use_tsc(use_tsc),
          use_distance_diff_consistency(use_distance_diff_consistency),
          distance_diff_thresh(distance_diff_thresh),
          use_second_scoring(use_second_scoring),
          use_degeneracy_filter(use_degeneracy_filter), refinement_iterations(refinement_iterations)
    {
        TORCH_CHECK(max_N >= 3 && num_pivot > 0 && k_neighbors > 0 && num_iterations > 0 && refinement_iterations >= 0,
                    "Invalid Sim3 integer options.");
        TORCH_CHECK(std::isfinite(epsilon_scale) && epsilon_scale > 0 && std::isfinite(tau_inlier) && tau_inlier > 0 &&
                    std::isfinite(radiu_nms) && radiu_nms >= 0 && std::isfinite(min_pair_dist) && min_pair_dist > 0 &&
                    std::isfinite(local_scale_min) && std::isfinite(local_scale_max) &&
                    local_scale_min > 0 && local_scale_max > local_scale_min &&
                    std::isfinite(local_scale_diff) && std::isfinite(distance_diff_thresh), "Invalid Sim3 thresholds.");
        if (this->local_scale_diff <= 0.0f)
        {
            this->local_scale_diff = this->epsilon_scale;
        }
        if (this->distance_diff_thresh <= 0.0f)
        {
            this->distance_diff_thresh = this->tau_inlier;
        }
        this->eval_metric = string_to_metric_type(metric_str);
    }

    static float median_of_k(std::vector<float> &vals)
    {
        if (vals.empty())
            return 0.0f;
        size_t mid = vals.size() / 2;
        std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
        float med = vals[mid];
        if (vals.size() % 2 == 0)
        {
            auto max_it = std::max_element(vals.begin(), vals.begin() + mid);
            med = 0.5f * (med + *max_it);
        }
        return med;
    }

    static float histogram_mode_scale(const std::vector<float> &scales,
                                      float log_min, float log_max,
                                      float bin_width)
    {
        if (scales.empty())
            return 1.0f;
        bin_width = std::max(bin_width, 1e-6f);
        std::unordered_map<int, std::pair<int, double>> bins;
        bins.reserve(scales.size());
        for (float s : scales)
        {
            if (s <= 0.0f)
                continue;
            float ls = std::log(s);
            if (ls < log_min || ls > log_max)
                continue;
            int bin = static_cast<int>(std::floor((ls - log_min) / bin_width));
            auto &entry = bins[bin];
            entry.first += 1;
            entry.second += static_cast<double>(ls);
        }
        if (bins.empty())
            return 1.0f;

        int best_count = -1;
        double best_sum = 0.0;
        int best_bin = std::numeric_limits<int>::max();
        for (const auto &kv : bins)
        {
            int count = kv.second.first;
            if (count > best_count || (count == best_count && kv.first < best_bin))
            {
                best_count = count;
                best_bin = kv.first;
                best_sum = kv.second.second;
            }
        }
        double mean_log = best_sum / static_cast<double>(std::max(best_count, 1));
        return static_cast<float>(std::exp(mean_log));
    }

    static torch::Tensor nondegenerate_clique_mask(const torch::Tensor &cliques_tensor,
                                                   const torch::Tensor &kpts_src,
                                                   const torch::Tensor &kpts_dst)
    {
        auto flat_indices = cliques_tensor.view({-1}).to(torch::kLong);
        int64_t num_cliques = cliques_tensor.size(0);

        auto src = kpts_src.index_select(0, flat_indices).view({num_cliques, 3, 3});
        auto dst = kpts_dst.index_select(0, flat_indices).view({num_cliques, 3, 3});

        auto src01 = src.select(1, 1) - src.select(1, 0);
        auto src02 = src.select(1, 2) - src.select(1, 0);
        auto src12 = src.select(1, 2) - src.select(1, 1);
        auto dst01 = dst.select(1, 1) - dst.select(1, 0);
        auto dst02 = dst.select(1, 2) - dst.select(1, 0);
        auto dst12 = dst.select(1, 2) - dst.select(1, 1);

        auto src_area2 = torch::norm(torch::cross(src01, src02, 1), 2, 1);
        auto dst_area2 = torch::norm(torch::cross(dst01, dst02, 1), 2, 1);

        auto src_max_edge = std::get<0>(torch::stack({
            torch::norm(src01, 2, 1),
            torch::norm(src02, 2, 1),
            torch::norm(src12, 2, 1)}, 1).max(1));
        auto dst_max_edge = std::get<0>(torch::stack({
            torch::norm(dst01, 2, 1),
            torch::norm(dst02, 2, 1),
            torch::norm(dst12, 2, 1)}, 1).max(1));

        const float rel_area_thresh = 1e-4f;
        auto src_thresh = rel_area_thresh * src_max_edge.pow(2);
        auto dst_thresh = rel_area_thresh * dst_max_edge.pow(2);
        return (src_area2 > src_thresh) & (dst_area2 > dst_thresh);
    }

    torch::Tensor ScaleTurboRegGPU::runRegTensor(torch::Tensor &kpts_src, torch::Tensor &kpts_dst)
    {
        torch::NoGradGuard no_grad;
        TORCH_CHECK(kpts_src.dim() == 2 && kpts_dst.sizes() == kpts_src.sizes() &&
                    kpts_src.size(1) == 3 && kpts_src.size(0) >= 3,
                    "Sim3 expects matching (N>=3, 3) correspondence tensors.");
        TORCH_CHECK(kpts_src.device() == kpts_dst.device(), "Inputs must share a device.");
        TORCH_CHECK(kpts_src.is_floating_point() && kpts_dst.is_floating_point(), "Inputs must be floating point.");
        TORCH_CHECK((torch::isfinite(kpts_src).all() & torch::isfinite(kpts_dst).all()).item<bool>(),
                    "Inputs must contain only finite values.");
        kpts_src = kpts_src.to(torch::kFloat32).contiguous();
        kpts_dst = kpts_dst.to(torch::kFloat32).contiguous();
        detail::normalize_pair(kpts_src, kpts_dst, max_N);

        auto run_core = [&]() {
            torch::Tensor best_trans;
            torch::Tensor kpts_src_iter = kpts_src;
            torch::Tensor kpts_dst_iter = kpts_dst;

            for (int iter = 0; iter < num_iterations; ++iter)
            {
                int64_t N_iter = kpts_src_iter.size(0);
                if (N_iter < 3)
                {
                    break;
                }

                torch::Tensor cliques_tensor;
                // Same dense compatibility graph on CPU and CUDA.

                {
                    auto dist_src = torch::cdist(kpts_src_iter, kpts_src_iter);  // (N, N)
                    auto dist_dst = torch::cdist(kpts_dst_iter, kpts_dst_iter);  // (N, N)

                    int64_t k_scale = std::min<int64_t>(k_neighbors, N_iter - 1);
                    if (k_scale <= 0)
                    {
                        break;
                    }

                    auto topk_src = torch::topk(dist_src.masked_fill(torch::eye(N_iter, dist_src.options()).to(torch::kBool), INFINITY), k_scale, /*dim=*/1, /*largest=*/false);
                    auto dists_src_knn = std::get<0>(topk_src).contiguous();
                    auto idx_knn_src = std::get<1>(topk_src).contiguous();

                    auto topk_dst = torch::topk(dist_dst.masked_fill(torch::eye(N_iter, dist_dst.options()).to(torch::kBool), INFINITY), k_scale, /*dim=*/1, /*largest=*/false);
                    auto dists_dst_knn = std::get<0>(topk_dst).contiguous();
                    auto idx_knn_dst = std::get<1>(topk_dst).contiguous();

                    torch::Tensor C2;
#ifdef TURBOREG_WITH_CUDA_KERNELS
                    if (dist_src.is_cuda() && k_scale <= 128) {
                        auto local_scales = detail::local_scales_cuda(dist_src, dist_dst, idx_knn_src, idx_knn_dst,
                            static_cast<int>(local_scale_mode), min_pair_dist, local_scale_min, local_scale_max, local_scale_diff);
                        C2 = detail::compatibility_cuda(dist_src, dist_dst, local_scales, local_scale_min, local_scale_max,
                            epsilon_scale, radiu_nms, use_tsc, use_distance_diff_consistency, distance_diff_thresh);
                    } else
#endif
                    {
                    auto dists_src_knn_cpu = dists_src_knn.to(torch::kCPU);
                    auto dists_dst_knn_cpu = dists_dst_knn.to(torch::kCPU);
                    auto acc_src = dists_src_knn_cpu.accessor<float, 2>();
                    auto acc_dst = dists_dst_knn_cpu.accessor<float, 2>();

                    float scale_min = std::max(local_scale_min, 1e-6f);
                    float scale_max = std::max(local_scale_max, scale_min * 1.01f);
                    float log_min = std::log(scale_min);
                    float log_max = std::log(scale_max);
                    float bin_width = std::max(local_scale_diff, 1e-3f);

                    std::vector<float> local_scales_vec(static_cast<size_t>(N_iter), 1.0f);
                    if (local_scale_mode == ScaleTurboRegGPU::LocalScaleMode::Median)
                    {
                        for (int64_t i = 0; i < N_iter; ++i)
                        {
                            std::vector<float> src_vals;
                            std::vector<float> dst_vals;
                            src_vals.reserve(static_cast<size_t>(k_scale));
                            dst_vals.reserve(static_cast<size_t>(k_scale));
                            for (int64_t j = 0; j < k_scale; ++j)
                            {
                                float ds = acc_src[i][j];
                                float dd = acc_dst[i][j];
                                if (ds > min_pair_dist)
                                    src_vals.push_back(ds);
                                if (dd > min_pair_dist)
                                    dst_vals.push_back(dd);
                            }
                            if (!src_vals.empty() && !dst_vals.empty())
                            {
                                float med_src = median_of_k(src_vals);
                                float med_dst = median_of_k(dst_vals);
                                local_scales_vec[static_cast<size_t>(i)] = med_dst / (med_src + 1e-8f);
                            }
                        }
                    }
                    else
                    {
                        auto row_indices = torch::arange(N_iter, idx_knn_src.options()).unsqueeze(1).expand_as(idx_knn_src);
                        auto dists_dst_corresponding = dist_dst.index({row_indices.reshape(-1), idx_knn_src.reshape(-1)})
                                                            .view({N_iter, k_scale});

                        auto row_indices_dst = torch::arange(N_iter, idx_knn_dst.options()).unsqueeze(1).expand_as(idx_knn_dst);
                        auto dists_src_for_dst_knn = dist_src.index({row_indices_dst.reshape(-1), idx_knn_dst.reshape(-1)})
                                                         .view({N_iter, k_scale});

                        auto dists_dst_corresponding_cpu = dists_dst_corresponding.to(torch::kCPU);
                        auto dists_src_for_dst_knn_cpu = dists_src_for_dst_knn.to(torch::kCPU);
                        auto idx_knn_src_cpu = idx_knn_src.to(torch::kCPU);
                        auto idx_knn_dst_cpu = idx_knn_dst.to(torch::kCPU);

                        auto acc_dst_corr = dists_dst_corresponding_cpu.accessor<float, 2>();
                        auto acc_src_for_dst = dists_src_for_dst_knn_cpu.accessor<float, 2>();
                        auto acc_idx_src = idx_knn_src_cpu.accessor<int64_t, 2>();
                        auto acc_idx_dst = idx_knn_dst_cpu.accessor<int64_t, 2>();

                        for (int64_t i = 0; i < N_iter; ++i)
                        {
                            std::unordered_map<int, float> scale_map;
                            scale_map.reserve(static_cast<size_t>(k_scale * 2));
                            for (int64_t j = 0; j < k_scale; ++j)
                            {
                                float ds = acc_src[i][j];
                                float dd = acc_dst_corr[i][j];
                                if (ds > min_pair_dist && dd > min_pair_dist)
                                {
                                    float s = dd / (ds + 1e-8f);
                                    if (s > scale_min && s < scale_max)
                                    {
                                        int nbr = static_cast<int>(acc_idx_src[i][j]);
                                        scale_map.emplace(nbr, s);
                                    }
                                }
                            }
                            for (int64_t j = 0; j < k_scale; ++j)
                            {
                                float dd = acc_dst[i][j];
                                float ds = acc_src_for_dst[i][j];
                                if (ds > min_pair_dist && dd > min_pair_dist)
                                {
                                    float s = dd / (ds + 1e-8f);
                                    if (s > scale_min && s < scale_max)
                                    {
                                        int nbr = static_cast<int>(acc_idx_dst[i][j]);
                                        if (scale_map.find(nbr) == scale_map.end())
                                        {
                                            scale_map.emplace(nbr, s);
                                        }
                                    }
                                }
                            }
                            if (scale_map.empty())
                            {
                                local_scales_vec[static_cast<size_t>(i)] = 1.0f;
                                continue;
                            }
                            std::vector<float> scales;
                            scales.reserve(scale_map.size());
                            for (const auto &kv : scale_map)
                            {
                                scales.push_back(kv.second);
                            }
                            if (local_scale_mode == ScaleTurboRegGPU::LocalScaleMode::UnionMedian)
                            {
                                local_scales_vec[static_cast<size_t>(i)] = median_of_k(scales);
                            }
                            else
                            {
                                local_scales_vec[static_cast<size_t>(i)] = histogram_mode_scale(scales, log_min, log_max, bin_width);
                            }
                        }
                    }
                    auto local_scales = torch::from_blob(local_scales_vec.data(), {N_iter}, torch::kFloat32).clone().to(kpts_src_iter.device());
                    auto log_local_scales = torch::log(local_scales + 1e-8f).contiguous();

                    auto edge_scales = dist_dst / (dist_src + 1e-8f);  // (N, N)
                    auto log_edge_scales = torch::log(edge_scales + 1e-8f);

                    auto log_scales_i = log_local_scales.unsqueeze(1);  // (N, 1)
                    auto log_scales_j = log_local_scales.unsqueeze(0);  // (1, N)

                    torch::Tensor scale_consistent;
                    if (use_distance_diff_consistency)
                    {
                        scale_consistent = torch::abs(dist_src - dist_dst) < distance_diff_thresh;
                    }
                    else if (use_tsc)
                    {
                        auto tri_max = torch::max(torch::max(log_scales_i, log_scales_j), log_edge_scales);
                        auto tri_min = torch::min(torch::min(log_scales_i, log_scales_j), log_edge_scales);
                        scale_consistent = (tri_max - tri_min) < epsilon_scale;
                    }
                    else
                    {
                        scale_consistent = torch::ones(
                            dist_src.sizes(),
                            torch::TensorOptions().dtype(torch::kBool).device(dist_src.device()));
                    }

                    auto mask_nms = (dist_src + dist_dst) <= radiu_nms;
                    auto mask_valid_scale = (edge_scales > local_scale_min) & (edge_scales < local_scale_max);
                    auto mask_valid_dist = dist_src > 1e-8f;

                    auto mask_local_range = (local_scales > local_scale_min) & (local_scales < local_scale_max);
                    auto mask_local_i = mask_local_range.unsqueeze(1);
                    auto mask_local_j = mask_local_range.unsqueeze(0);

                    torch::Tensor adjacency_mask;
                    if (!use_distance_diff_consistency && !use_tsc)
                    {
                        adjacency_mask = mask_valid_dist & (~mask_nms);
                    }
                    else
                    {
                        adjacency_mask = scale_consistent & mask_valid_scale & mask_valid_dist & (~mask_nms) & mask_local_i & mask_local_j;
                    }

                    C2 = adjacency_mask.to(torch::kFloat32);
                    }
                    auto score_matrix = use_second_scoring ? torch::matmul(C2, C2) * C2 : C2;

                    auto score_up = torch::triu(score_matrix, 1);
                    auto flat_score_up = score_up.flatten();
                    int64_t topk_num = std::min<int64_t>(num_pivot, flat_score_up.numel());
                    if (topk_num <= 0)
                    {
                        break;
                    }

                    auto topk_result = torch::topk(flat_score_up, topk_num);
                    auto idx_topk = std::get<1>(topk_result).index({std::get<0>(topk_result) > 0});
                    if (idx_topk.numel() == 0) break;

                    auto pivots = torch::stack({(idx_topk / N_iter).to(torch::kLong),
                                                (idx_topk % N_iter).to(torch::kLong)},
                                               1);

                    auto score_for_search = score_up.clone();
                    auto adjacency_up = torch::triu(C2, 1);
                    auto adjacency_pivot_0 = adjacency_up.index_select(0, pivots.select(1, 0)) > 0;
                    auto adjacency_pivot_1 = adjacency_up.index_select(0, pivots.select(1, 1)) > 0;
                    auto indic_c3_torch = adjacency_pivot_0 & adjacency_pivot_1;

                    auto pivot_scores = score_for_search.index({pivots.select(1, 0), pivots.select(1, 1)});
                    auto clique_scores = pivot_scores.unsqueeze(1) +
                                         (score_for_search.index_select(0, pivots.select(1, 0)) +
                                          score_for_search.index_select(0, pivots.select(1, 1)));
                    auto masked_clique_scores = clique_scores * indic_c3_torch.to(torch::kFloat32);

                    int64_t clique_per_pivot = 2;
                    auto topk_result_row = torch::topk(masked_clique_scores, /*k=*/clique_per_pivot, /*dim=*/1);
                    auto topk_K2 = std::get<1>(topk_result_row);

                    int64_t num_pivots = pivots.size(0);
                    cliques_tensor = torch::zeros({num_pivots * clique_per_pivot, 3}, torch::kInt32).to(kpts_src_iter.device());
                    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), torch::indexing::Slice(0, 2)}, pivots);
                    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 0}));
                    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), torch::indexing::Slice(0, 2)}, pivots);
                    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 1}));
                    auto valid_extensions = torch::cat({std::get<0>(topk_result_row).select(1, 0) > 0,
                                                        std::get<0>(topk_result_row).select(1, 1) > 0});
                    cliques_tensor = cliques_tensor.index({valid_extensions}).contiguous();
                }
                if (!cliques_tensor.defined() || cliques_tensor.numel() == 0)
                {
                    break;
                }
                if (use_degeneracy_filter)
                {
                    auto valid_clique_mask = nondegenerate_clique_mask(cliques_tensor, kpts_src_iter, kpts_dst_iter);
                    cliques_tensor = cliques_tensor.index({valid_clique_mask}).contiguous();
                    if (cliques_tensor.size(0) == 0)
                    {
                        break;
                    }
                }

                torch::Tensor cliques_wise_trans;
                ModelSelection model_selector(this->eval_metric, this->tau_inlier);
                verificationSim3Metric(cliques_tensor, kpts_src_iter, kpts_dst_iter, model_selector,
                                       best_trans, cliques_wise_trans);

                if (iter < num_iterations - 1 && best_trans.defined())
                {
                    auto sR = best_trans.slice(0, 0, 3).slice(1, 0, 3);
                    auto t = best_trans.slice(0, 0, 3).slice(1, 3, 4);
                    auto transformed = torch::matmul(kpts_src_iter, sR.t()) + t.t();

                    auto residuals = torch::norm(transformed - kpts_dst_iter, 2, 1);
                    auto inlier_mask = residuals < tau_inlier;

                    int inlier_count = inlier_mask.sum().item<int>();
                    if (inlier_count < 10)
                    {
                        break;
                    }

                    kpts_src_iter = kpts_src_iter.index({inlier_mask}).contiguous();
                    kpts_dst_iter = kpts_dst_iter.index({inlier_mask}).contiguous();
                }
            }

            if (!best_trans.defined())
            {
                throw std::runtime_error("ScaleTurboRegGPU failed to estimate transformation.");
            }

            torch::Tensor refined_trans = post_refinement_sim3(best_trans, kpts_src, kpts_dst, refinement_iterations, tau_inlier);
            return refined_trans;
        };

        return run_core();
    }

    SimilarityTransform ScaleTurboRegGPU::runRegCXX(torch::Tensor &kpts_src, torch::Tensor &kpts_dst)
    {
        return SimilarityTransform(runRegTensor(kpts_src, kpts_dst));
    }

    torch::Tensor ScaleTurboRegGPU::runRegCXXReturnTensor(torch::Tensor &kpts_src, torch::Tensor &kpts_dst)
    {
        auto T = runRegTensor(kpts_src, kpts_dst);
        // The tensor API need not decompose on the host and reconstruct the same matrix.
        // Preserve the legacy round trip on CPU for compatibility with archived results.
        return T.is_cuda() ? T : SimilarityTransform(T).getTransformation();
    }

    // Umeyama algorithm implementation
    // Reference: S. Umeyama, "Least-Squares Estimation of Transformation Parameters
    //            Between Two Point Patterns", IEEE PAMI, 1991
    torch::Tensor umeyama_transform_3d(const torch::Tensor &A, const torch::Tensor &B,
                                       const torch::Tensor &weights,
                                       float weight_threshold)
    {
        TORCH_CHECK(A.dim() == 3 && B.sizes() == A.sizes() && A.size(2) == 3 && A.size(1) >= 3,
                    "Umeyama expects matching (batch, N>=3, 3) tensors.");
#ifdef TURBOREG_WITH_CUDA_KERNELS
        if (A.is_cuda() && A.scalar_type() == torch::kFloat32 && A.size(1) <= 256)
            return detail::umeyama_cuda(A, B, weights, weight_threshold);
#endif
        auto A_in = A;
        auto B_in = B;
        int64_t bs = A_in.size(0);
        int64_t n = A_in.size(1); // Number of points

        torch::Tensor W = weights;
        if (!weights.defined() || weights.numel() == 0)
        {
            W = torch::ones({bs, n}, A.options());
        }
        else
        {
            // Ensure W has shape (bs, n)
            if (W.dim() == 2 && W.size(0) == bs && W.size(1) == n)
            {
                // Already correct shape
            }
            else if (W.dim() == 1 && W.size(0) == n)
            {
                W = W.unsqueeze(0).expand({bs, n});
            }
        }
        W = W.masked_fill(W < weight_threshold, 0);

        // Batched Umeyama (B, N, 3)
        auto W_sum = W.sum(/*dim=*/1, /*keepdim=*/true) + 1e-8f; // (B,1)
        auto W_sum_exp = W_sum.unsqueeze(-1);                   // (B,1,1)
        auto centroid_A = (A_in * W.unsqueeze(-1)).sum(1, true) / W_sum_exp; // (B,1,3)
        auto centroid_B = (B_in * W.unsqueeze(-1)).sum(1, true) / W_sum_exp; // (B,1,3)

        auto Am = A_in - centroid_A; // (B,N,3)
        auto Bm = B_in - centroid_B; // (B,N,3)

        auto Am_weighted = Am * W.unsqueeze(-1);                      // (B,N,3)
        auto H = torch::bmm(Am_weighted.transpose(1, 2), Bm) / W_sum_exp;  // (B,3,3)
        auto var_A = (Am.pow(2).sum(-1) * W).sum(1) / W_sum.squeeze(1); // (B)

        auto svd = torch::linalg_svd(H);
        auto U = std::get<0>(svd);   // (B,3,3)
        auto S = std::get<1>(svd);   // (B,3)
        auto Vh = std::get<2>(svd);  // (B,3,3)
        auto V = Vh.transpose(1, 2); // (B,3,3)

        auto det = torch::sign(torch::det(torch::bmm(V, U.transpose(1, 2)))); // (B)
        auto D = torch::eye(3, A.options()).unsqueeze(0).repeat({bs, 1, 1});
        D.index_put_({torch::indexing::Slice(), 2, 2}, det);
        auto R = torch::bmm(torch::bmm(V, D), U.transpose(1, 2)); // (B,3,3)

        auto d_diag = torch::ones({bs, 3}, A.options());
        d_diag.index_put_({torch::indexing::Slice(), 2}, det);
        auto trace_DS = (d_diag * S).sum(1); // (B)
        auto scale = trace_DS / var_A.clamp_min(1e-12f);

        auto centroid_A_vec = centroid_A.squeeze(1); // (B,3)
        auto centroid_B_vec = centroid_B.squeeze(1); // (B,3)
        auto t = centroid_B_vec - (scale.unsqueeze(1) * torch::bmm(R, centroid_A_vec.unsqueeze(-1)).squeeze(-1)); // (B,3)

        auto trans = torch::eye(4, A.options()).unsqueeze(0).repeat({bs, 1, 1});
        trans.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)},
                         scale.view({bs, 1, 1}) * R);
        trans.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 3}, t);

        return trans;
    }

    // Helper function to transform points with Sim-3
    torch::Tensor transformSim3(const torch::Tensor &src_keypts, const torch::Tensor &trans)
    {
        // trans: 4x4 with scale encoded in rotation part
        auto sR = trans.narrow(0, 0, 3).narrow(1, 0, 3); // 3x3 (includes scale)
        auto t = trans.narrow(0, 0, 3).narrow(1, 3, 1);  // 3x1
        return torch::matmul(src_keypts, sR.t()) + t.t();
    }

    torch::Tensor post_refinement_sim3(torch::Tensor initial_trans,
                                       torch::Tensor src_keypts,
                                       torch::Tensor tgt_keypts,
                                       int it_num,
                                       float inlier_threshold)
    {
        torch::Tensor inlier_threshold_list = torch::full({it_num}, inlier_threshold, initial_trans.options());

        torch::Tensor previous_mask;
        for (int i = 0; i < it_num; ++i)
        {
            // Transform source points
            torch::Tensor warped_src_keypts = transformSim3(src_keypts, initial_trans);

            // Calculate L2 distance
            torch::Tensor L2_dis = torch::norm(warped_src_keypts - tgt_keypts, 2, 1);

            // Find inliers
            auto pred_inlier = L2_dis < inlier_threshold_list[i];
            int inlier_num = pred_inlier.sum().item<int>();

            if (previous_mask.defined() && torch::equal(pred_inlier, previous_mask))
            {
                break;
            }
            previous_mask = pred_inlier.clone();

            if (inlier_num < 3)
            {
                break;
            }

            // Compute weights
            auto weight = 1.0f / (1.0f + torch::pow(L2_dis.index({pred_inlier}) / inlier_threshold_list[i], 2));

            // Refine transformation using Umeyama
            initial_trans = umeyama_transform_3d(
                                src_keypts.index({pred_inlier}).unsqueeze(0),
                                tgt_keypts.index({pred_inlier}).unsqueeze(0),
                                weight.unsqueeze(0))
                                .squeeze(0);
        }

        return initial_trans;
    }

    void verificationSim3Metric(const torch::Tensor &cliques_tensor,
                                const torch::Tensor &kpts_src,
                                const torch::Tensor &kpts_dst,
                                ModelSelection &model_selector,
                                torch::Tensor &best_trans,
                                torch::Tensor &cliques_wise_trans)
    {
        // Extract 3-point sets for each clique
        // cliques_tensor: (num_cliques, 3) containing indices
        auto flat_indices = cliques_tensor.view({-1}).to(torch::kLong);
        int64_t num_cliques = cliques_tensor.size(0);

        torch::Tensor kpts_src_sub = kpts_src.index_select(0, flat_indices).view({num_cliques, 3, 3});
        torch::Tensor kpts_dst_sub = kpts_dst.index_select(0, flat_indices).view({num_cliques, 3, 3});

        // Guard against unexpected (C, 3, N) layout
        if (kpts_src_sub.size(2) != 3 && kpts_src_sub.size(1) == 3)
        {
            kpts_src_sub = kpts_src_sub.transpose(1, 2).contiguous();
        }
        if (kpts_dst_sub.size(2) != 3 && kpts_dst_sub.size(1) == 3)
        {
            kpts_dst_sub = kpts_dst_sub.transpose(1, 2).contiguous();
        }

        // Use batched Umeyama to estimate Sim-3 transformations for all cliques.
        // kpts_*_sub shapes are (C, 3, 3) -> batch size C, N=3.
        cliques_wise_trans = umeyama_transform_3d(
            kpts_src_sub,
            kpts_dst_sub,
            torch::Tensor(),
            0.0f);

        // Select best transformation using model selector
        torch::Tensor idx_best_guess = model_selector.calculate_best_clique(cliques_wise_trans, kpts_src, kpts_dst);

        best_trans = cliques_wise_trans.index_select(0, idx_best_guess.reshape({1})).squeeze(0);
    }

}

#include <turboreg/turboreg.hpp>
#include <turboreg/core_turboreg_gpu.hpp>
#include <turboreg/detail/registration_utils.hpp>
#include <turboreg/utils_debug.hpp>

using namespace turboreg;

TurboRegGPU::TurboRegGPU(int max_N, float tau_length_consis, int num_pivot, float radiu_nms, float tau_inlier, const std::string &metric_str)
    : max_N(max_N), tau_length_consis(tau_length_consis), radiu_nms(radiu_nms), num_pivot(num_pivot), tau_inlier(tau_inlier)
{
    this->eval_metric = string_to_metric_type(metric_str);
}

RigidTransform TurboRegGPU::runRegCXX(torch::Tensor &kpts_src, torch::Tensor &kpts_dst)
{
    detail::normalize_pair(kpts_src, kpts_dst, max_N);
    const auto device = kpts_src.device();
    const auto N_node = static_cast<int>(kpts_src.size(0));
    if (N_node < 3)
    {
        throw std::runtime_error("TurboRegGPU requires at least 3 correspondences.");
    }

    auto src_dist = torch::norm(kpts_src.unsqueeze(1) - kpts_src.unsqueeze(0), 2, -1);
    auto target_dist = torch::norm(kpts_dst.unsqueeze(1) - kpts_dst.unsqueeze(0), 2, -1);
    auto cross_dist = torch::abs(src_dist - target_dist);
    torch::Tensor C2;

    if (!hard)
    {
        C2 = torch::relu(1 - torch::pow(cross_dist / tau_length_consis, 2));
    }
    else
    {
        C2 = (cross_dist < tau_length_consis).to(torch::kFloat32);
    }

    auto mask = (src_dist + target_dist) <= radiu_nms;
    C2.masked_fill_(mask, 0);

    auto SC2 = torch::matmul(C2, C2) * C2;

    auto SC2_up = torch::triu(SC2, 1);
    auto flat_SC2_up = SC2_up.flatten();
    const auto topk_num = std::min<int64_t>(num_pivot, flat_SC2_up.numel());
    if (topk_num <= 0)
    {
        throw std::runtime_error("TurboRegGPU did not find a valid pivot.");
    }

    auto topk_result = torch::topk(flat_SC2_up, topk_num);
    auto idx_topk = std::get<1>(topk_result);

    auto pivots = torch::stack({(idx_topk / N_node).to(torch::kLong),
                                (idx_topk % N_node).to(torch::kLong)},
                               1);

    auto SC2_for_search = SC2_up.clone();

    auto SC2_pivot_0 = SC2_for_search.index_select(0, pivots.select(1, 0)) > 0;
    auto SC2_pivot_1 = SC2_for_search.index_select(0, pivots.select(1, 1)) > 0;
    auto indic_c3_torch = SC2_pivot_0 & SC2_pivot_1;

    auto SC2_pivots = SC2_for_search.index({pivots.select(1, 0), pivots.select(1, 1)});

    auto SC2_ADD_C3 = SC2_pivots.unsqueeze(1) +
                      (SC2_for_search.index_select(0, pivots.select(1, 0)) +
                       SC2_for_search.index_select(0, pivots.select(1, 1)));

    auto SC2_C3 = SC2_ADD_C3 * indic_c3_torch.to(torch::kFloat32);

    auto topk_result_row = torch::topk(SC2_C3, /*k=*/2, /*dim=*/1);
    auto topk_K2 = std::get<1>(topk_result_row);

    int num_pivots = pivots.size(0);
    auto cliques_tensor = torch::zeros({num_pivots * 2, 3},
                                       torch::TensorOptions().dtype(torch::kInt32).device(device));

    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), torch::indexing::Slice(0, 2)}, pivots);
    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 0}));

    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), torch::indexing::Slice(0, 2)}, pivots);
    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 1}));

    torch::Tensor best_in_num, best_trans, res, cliques_wise_trans, cliquewise_in_num;
    ModelSelection model_selector(this->eval_metric, this->tau_inlier);
    verificationV2Metric(cliques_tensor, kpts_src, kpts_dst, model_selector,
                         best_in_num, best_trans, res,
                         cliques_wise_trans);

    torch::Tensor refined_trans = post_refinement(best_trans, kpts_src, kpts_dst, 20, this->tau_inlier);
    RigidTransform trans_final(refined_trans);
    return trans_final;
}

torch::Tensor TurboRegGPU::runRegCXXReturnTensor(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all)
{
    return runRegCXX(kpts_src_all, kpts_dst_all).getTransformation();
}

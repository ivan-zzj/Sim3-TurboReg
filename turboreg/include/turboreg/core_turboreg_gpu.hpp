#pragma once

#include <torch/torch.h>
#include "model_selection.hpp"

namespace turboreg
{

    torch::Tensor rigid_transform_3d(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &weights, float weight_threshold);
    torch::Tensor post_refinement(torch::Tensor initial_trans, torch::Tensor src_keypts, torch::Tensor tgt_keypts, int it_num, float inlier_threshold);
    void verification(const torch::Tensor &cliques_tensor,
                      const torch::Tensor &kpts_src,
                      const torch::Tensor &kpts_dst,
                      float inlier_threshold,
                      torch::Tensor &best_in_num,
                      torch::Tensor &best_in_indic,
                      torch::Tensor &best_trans,
                      torch::Tensor &res,
                      torch::Tensor &cliques_wise_trans,
                      torch::Tensor &cliquewise_in_num);
    void verificationV2Metric(const torch::Tensor &cliques_tensor,
                              const torch::Tensor &kpts_src,
                              const torch::Tensor &kpts_dst,
                              ModelSelection &model_selector,
                              torch::Tensor &best_in_num,
                              torch::Tensor &best_trans,
                              torch::Tensor &res,
                              torch::Tensor &cliques_wise_trans);
}
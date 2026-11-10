#ifndef SCALE_TURBOREG_HPP
#define SCALE_TURBOREG_HPP

#include <torch/torch.h>
#include <vector>
#include "similarity_transform.hpp"
#include "model_selection.hpp"

namespace turboreg
{

    class ScaleTurboRegGPU
    {
    public:
        enum class LocalScaleMode
        {
            Median = 0,
            Histogram = 1,
            UnionMedian = 2
        };
        ScaleTurboRegGPU(int max_N, float epsilon_scale, int num_pivot, float radiu_nms,
                         float tau_inlier, const std::string &metric_str,
                         int k_neighbors = 20, int num_iterations = 2,
                         float trim_ratio = 0.1f, float min_pair_dist = 1e-3f,
                         float local_scale_min = 0.1f, float local_scale_max = 10.0f,
                         float local_scale_diff = -1.0f,
                         const std::string &local_scale_mode = "hist",
                         bool use_tsc = true,
                         bool use_distance_diff_consistency = false,
                         float distance_diff_thresh = -1.0f,
                         bool use_second_scoring = true,
                         bool use_degeneracy_filter = true,
                         int refinement_iterations = 20);

        SimilarityTransform runRegCXX(torch::Tensor &kpts_src, torch::Tensor &kpts_dst);
        torch::Tensor runRegCXXReturnTensor(torch::Tensor &kpts_src, torch::Tensor &kpts_dst);

    private:

        torch::Tensor runRegTensor(torch::Tensor &src, torch::Tensor &dst);

        int max_N;
        float epsilon_scale;
        float radiu_nms;
        int num_pivot;
        float tau_inlier;
        int k_neighbors;
        int num_iterations;

        float trim_ratio;
        float min_pair_dist;
        float local_scale_min;
        float local_scale_max;
        float local_scale_diff;
        LocalScaleMode local_scale_mode;
        bool use_tsc;
        bool use_distance_diff_consistency;
        float distance_diff_thresh;
        bool use_second_scoring;
        bool use_degeneracy_filter;
        int refinement_iterations;
        MetricType eval_metric;
    };

    torch::Tensor umeyama_transform_3d(const torch::Tensor &A, const torch::Tensor &B,
                                       const torch::Tensor &weights = torch::Tensor(),
                                       float weight_threshold = 0.0f);

    torch::Tensor post_refinement_sim3(torch::Tensor initial_trans,
                                        torch::Tensor src_keypts,
                                        torch::Tensor tgt_keypts,
                                        int it_num,
                                        float inlier_threshold);

    void verificationSim3Metric(const torch::Tensor &cliques_tensor,
                                 const torch::Tensor &kpts_src,
                                 const torch::Tensor &kpts_dst,
                                 ModelSelection &model_selector,
                                 torch::Tensor &best_trans,
                                 torch::Tensor &cliques_wise_trans);

}

#endif // SCALE_TURBOREG_HPP

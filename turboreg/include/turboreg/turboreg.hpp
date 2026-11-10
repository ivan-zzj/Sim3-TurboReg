#ifndef TURBOREG_HPP
#define TURBOREG_HPP

#include <torch/torch.h>
#include <Eigen/Dense>
#include <vector>
#include "model_selection.hpp"
#include "rigid_transform.hpp"

namespace turboreg
{

    class TurboRegGPU
    {
    public:
        TurboRegGPU(int max_N, float tau_length_consis, int num_pivot, float radiu_nms, float tau_inlier, const std::string &metric_str);
        RigidTransform runRegCXX(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all);
        torch::Tensor runRegCXXReturnTensor(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all);

    private:
        int max_N;
        float tau_length_consis;
        float radiu_nms;
        int num_pivot;
        float tau_inlier;
        bool hard = true;
        MetricType eval_metric;
    };

}

#endif // TURBOREG_HPP

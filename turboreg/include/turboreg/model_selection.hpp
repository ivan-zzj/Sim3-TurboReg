#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace turboreg {

enum class MetricType {
    INLIER_COUNT,
    MAE,
    MSE
};

MetricType string_to_metric_type(const std::string& metric_str);

class ModelSelection {
public:
    ModelSelection(const std::string& metric_str, float inlier_threshold);
    ModelSelection(MetricType metric, float inlier_threshold);

    torch::Tensor calculate_best_clique(
        const torch::Tensor& cliques_wise_trans, 
        const torch::Tensor& kpts_src, 
        const torch::Tensor& kpts_dst);

private:
    MetricType metric_type;
    float inlier_threshold;
};

}


#endif // MODEL_SELECTION_H

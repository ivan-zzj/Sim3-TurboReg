#include <turboreg/model_selection.hpp>
#include <torch/torch.h>
#include <unordered_map>
#include <stdexcept>
#include <turboreg/utils_debug.hpp>

turboreg::MetricType turboreg::string_to_metric_type(const std::string &metric_str)
{
    static const std::unordered_map<std::string, MetricType> string_to_enum = {
        {"IN", MetricType::INLIER_COUNT},
        {"MAE", MetricType::MAE},
        {"MSE", MetricType::MSE}};

    auto it = string_to_enum.find(metric_str);
    if (it == string_to_enum.end())
    {
        throw std::invalid_argument("Invalid metric type string: " + metric_str);
    }
    return it->second;
}

turboreg::ModelSelection::ModelSelection(const std::string &metric_str, float inlier_threshold)
    : metric_type(string_to_metric_type(metric_str)), inlier_threshold(inlier_threshold) {}

turboreg::ModelSelection::ModelSelection(MetricType metric, float inlier_threshold) : metric_type(metric), inlier_threshold(inlier_threshold) {}

torch::Tensor turboreg::ModelSelection::calculate_best_clique(
    const torch::Tensor &cliques_wise_trans,
    const torch::Tensor &kpts_src,
    const torch::Tensor &kpts_dst)
{
    TORCH_CHECK(cliques_wise_trans.size(0) > 0, "No hypotheses to score.");
    // Bound verification memory: never materialize all hypotheses x N x 3.
    std::vector<torch::Tensor> scores;
    for (int64_t start = 0; start < cliques_wise_trans.size(0); start += 128) {
        auto models = cliques_wise_trans.slice(0, start, start + 128);
        auto rotations = models.slice(1, 0, 3).slice(2, 0, 3);
        auto translations = models.slice(1, 0, 3).slice(2, 3, 4);
        auto predicted = torch::matmul(rotations, kpts_src.t()) + translations;
        auto residual = (predicted.transpose(1, 2) - kpts_dst.unsqueeze(0)).norm(2, -1);
        torch::Tensor score;
        if (metric_type == MetricType::INLIER_COUNT) {
            score = (residual < inlier_threshold).sum(-1).to(torch::kFloat32);
        } else {
            auto weights = (1 - residual / inlier_threshold).clamp_min(0);
            score = (metric_type == MetricType::MSE ? weights.square() : weights).sum(-1);
        }
        score = score.masked_fill(~torch::isfinite(models).all(2).all(1), -1);
        scores.push_back(score);
    }
    return torch::cat(scores).argmax();
}

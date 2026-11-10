#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace turboreg::detail
{

struct CpuKnnResult
{
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<float>> dists;
};

inline torch::Tensor normalize_points(torch::Tensor points)
{
    if (points.dim() == 3 && points.size(0) == 1)
    {
        points = points.squeeze(0);
    }
    if (points.dim() == 2 && points.size(1) != 3 && points.size(0) == 3)
    {
        points = points.transpose(0, 1);
    }
    return points.contiguous();
}

inline void normalize_pair(torch::Tensor &src, torch::Tensor &dst, int64_t max_points)
{
    src = normalize_points(src);
    dst = normalize_points(dst);

    if (src.dim() != 2 || dst.dim() != 2 || src.size(1) != 3 || dst.size(1) != 3)
    {
        throw std::runtime_error("TurboReg expects inputs with shape (N, 3).");
    }

    const auto point_count = std::min<int64_t>({src.size(0), dst.size(0), max_points});
    if (point_count < src.size(0))
    {
        src = src.slice(0, 0, point_count);
    }
    if (point_count < dst.size(0))
    {
        dst = dst.slice(0, 0, point_count);
    }
}

inline CpuKnnResult knn_bruteforce_cpu(const torch::Tensor &points_cpu, int k)
{
    auto points = points_cpu.to(torch::kCPU).contiguous();
    if (points.dim() != 2 || points.size(1) != 3)
    {
        throw std::runtime_error("knn_bruteforce_cpu expects a tensor with shape (N, 3).");
    }

    const int64_t point_count = points.size(0);
    CpuKnnResult result;
    result.indices.resize(static_cast<size_t>(point_count));
    result.dists.resize(static_cast<size_t>(point_count));
    if (point_count <= 1 || k <= 0)
    {
        return result;
    }

    k = std::min<int64_t>(k, point_count - 1);
    auto acc = points.accessor<float, 2>();
    for (int64_t i = 0; i < point_count; ++i)
    {
        std::vector<std::pair<float, int>> dist_idx;
        dist_idx.reserve(static_cast<size_t>(point_count - 1));
        for (int64_t j = 0; j < point_count; ++j)
        {
            if (i == j)
            {
                continue;
            }

            const float dx = acc[i][0] - acc[j][0];
            const float dy = acc[i][1] - acc[j][1];
            const float dz = acc[i][2] - acc[j][2];
            dist_idx.emplace_back(std::sqrt(dx * dx + dy * dy + dz * dz), static_cast<int>(j));
        }

        if (static_cast<int64_t>(dist_idx.size()) > k)
        {
            std::nth_element(
                dist_idx.begin(),
                dist_idx.begin() + k,
                dist_idx.end(),
                [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
            dist_idx.resize(static_cast<size_t>(k));
        }

        std::sort(
            dist_idx.begin(),
            dist_idx.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

        auto &row_indices = result.indices[static_cast<size_t>(i)];
        auto &row_dists = result.dists[static_cast<size_t>(i)];
        row_indices.reserve(dist_idx.size());
        row_dists.reserve(dist_idx.size());
        for (const auto &entry : dist_idx)
        {
            row_indices.push_back(entry.second);
            row_dists.push_back(entry.first);
        }
    }

    return result;
}

inline int64_t edge_key(int i, int j)
{
    if (i > j)
    {
        std::swap(i, j);
    }
    return (static_cast<int64_t>(i) << 32) | static_cast<uint32_t>(j);
}

inline int get_edge_score(const std::unordered_map<int64_t, int> &edge_scores, int i, int j)
{
    const auto it = edge_scores.find(edge_key(i, j));
    if (it == edge_scores.end())
    {
        return 0;
    }
    return it->second;
}

inline int count_common_neighbors(const std::vector<int> &lhs, const std::vector<int> &rhs)
{
    int count = 0;
    size_t i = 0;
    size_t j = 0;
    while (i < lhs.size() && j < rhs.size())
    {
        if (lhs[i] == rhs[j])
        {
            ++count;
            ++i;
            ++j;
        }
        else if (lhs[i] < rhs[j])
        {
            ++i;
        }
        else
        {
            ++j;
        }
    }
    return count;
}

} // namespace turboreg::detail

#pragma once
#include <ATen/ATen.h>

namespace turboreg::detail {
// Supported for k <= 128; callers keep the general CPU fallback for larger k.
at::Tensor local_scales_cuda(const at::Tensor& ds, const at::Tensor& dd,
                            const at::Tensor& src_ids, const at::Tensor& dst_ids,
                            int mode, float min_distance, float scale_min,
                            float scale_max, float bin_width);
at::Tensor compatibility_cuda(const at::Tensor& ds, const at::Tensor& dd,
                             const at::Tensor& scales, float scale_min,
                             float scale_max, float epsilon, float radius,
                             bool tsc, bool distance_diff, float distance_threshold);
at::Tensor umeyama_cuda(const at::Tensor& a, const at::Tensor& b,
                       const at::Tensor& weights, float weight_threshold);
}

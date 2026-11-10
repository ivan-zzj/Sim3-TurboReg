#ifndef SIMILARITY_TRANSFORM_HPP
#define SIMILARITY_TRANSFORM_HPP

#include <torch/torch.h>

namespace turboreg
{

    class SimilarityTransform
    {
    public:
        SimilarityTransform();
        explicit SimilarityTransform(const torch::Tensor &trans);
        SimilarityTransform(const torch::Tensor &rotation, const torch::Tensor &translation, float scale);
        void copy(const SimilarityTransform &other);
        torch::Tensor getTransformation() const;
        torch::Tensor getRotation() const;
        torch::Tensor getTranslation() const;
        float getScale() const;
        torch::Tensor transformPoints(const torch::Tensor &pts) const;
        int countInliers(const torch::Tensor &kptsSrc, const torch::Tensor &kptsDst, float metricThresh) const;

    public:
        torch::Tensor R;
        torch::Tensor t;
        float s;
    };

}

#endif // SIMILARITY_TRANSFORM_HPP

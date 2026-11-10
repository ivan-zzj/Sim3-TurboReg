#include <turboreg/similarity_transform.hpp>

namespace turboreg
{

    SimilarityTransform::SimilarityTransform()
        : s(1.0f)
    {
        R = torch::eye(3, torch::kFloat32);
        t = torch::zeros({3, 1}, torch::kFloat32);
    }

    SimilarityTransform::SimilarityTransform(const torch::Tensor &trans)
    {
        auto sR = trans.slice(0, 0, 3).slice(1, 0, 3);

        s = torch::norm(sR.slice(1, 0, 1)).item<float>();

        if (s > 1e-6f)
        {
            R = sR / s;
        }
        else
        {
            R = torch::eye(3, trans.options());
            s = 1.0f;
        }

        t = trans.slice(0, 0, 3).slice(1, 3, 4).clone();
    }

    SimilarityTransform::SimilarityTransform(const torch::Tensor &rotation, const torch::Tensor &translation, float scale)
        : R(rotation.clone()), t(translation.clone()), s(scale)
    {
    }

    void SimilarityTransform::copy(const SimilarityTransform &other)
    {
        R = other.R.clone();
        t = other.t.clone();
        s = other.s;
    }

    torch::Tensor SimilarityTransform::getTransformation() const
    {
        torch::Tensor trans = torch::eye(4, R.options());
        trans.slice(0, 0, 3).slice(1, 0, 3) = s * R;
        trans.slice(0, 0, 3).slice(1, 3, 4) = t;
        return trans;
    }

    torch::Tensor SimilarityTransform::getRotation() const
    {
        return R.clone();
    }

    torch::Tensor SimilarityTransform::getTranslation() const
    {
        return t.clone();
    }

    float SimilarityTransform::getScale() const
    {
        return s;
    }

    torch::Tensor SimilarityTransform::transformPoints(const torch::Tensor &pts) const
    {
        auto R_dev = R.to(pts.device());
        auto t_dev = t.to(pts.device());
        return s * torch::matmul(pts, R_dev.t()) + t_dev.t();
    }

    int SimilarityTransform::countInliers(const torch::Tensor &kptsSrc, const torch::Tensor &kptsDst, float metricThresh) const
    {
        auto transformed = transformPoints(kptsSrc);
        auto dist = torch::norm(transformed - kptsDst, 2, 1);
        return (dist < metricThresh).sum().item<int>();
    }

}

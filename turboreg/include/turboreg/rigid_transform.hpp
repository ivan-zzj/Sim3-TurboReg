#ifndef RIGID_TRANSFORM_HPP
#define RIGID_TRANSFORM_HPP

#include <torch/torch.h>
#include <iostream>

namespace turboreg
{

    class RigidTransform
    {
    public:
        RigidTransform();

        explicit RigidTransform(const torch::Tensor &trans);

        void copy(const torch::Tensor &R_in, const torch::Tensor &t_in);

        void copy(const RigidTransform &other);

        RigidTransform(const torch::Tensor &rotation, const torch::Tensor &translation);

        torch::Tensor getTransformation() const;

        torch::Tensor getRotation() const;

        torch::Tensor getTranslation() const;

        torch::Tensor transformPoints(const torch::Tensor &pts) const;

        int countInliers(const torch::Tensor &kptsSrc, const torch::Tensor &kptsDst, float metricThresh) const;

    public:
        torch::Tensor R; // Rotation matrix (3x3)
        torch::Tensor t; // Translation vector (3x1)
    };

}

#endif // RIGID_TRANSFORM_HPP

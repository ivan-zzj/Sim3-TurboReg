#ifndef UTILS_PCR_HPP
#define UTILS_PCR_HPP

#include <torch/torch.h>

namespace turboreg {

double calculateRotationError(const torch::Tensor &est, const torch::Tensor &gt);
double calculateTranslationError(const torch::Tensor &est, const torch::Tensor &gt);
bool evaluationEst(const torch::Tensor &est,
                   const torch::Tensor &gt,
                   double reThresh,
                   double teThresh,
                   double &RE,
                   double &TE);

}
#endif // UTILS_PCR_HPP
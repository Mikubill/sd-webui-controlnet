#include <cuda_runtime_api.h>
#include <torch/script.h>
// clang-format off
// TODO: make spconv_utils.h order agnostic
#include "../spconv_utils.h"
// clang-format on
#include <utils/spconv/spconv/maxpool.h>

#include "pytorch_cuda_helper.hpp"

torch::Tensor IndiceMaxpoolForwardCUDAKernelLauncher(torch::Tensor features,
                                                     torch::Tensor indicePairs,
                                                     torch::Tensor indiceNum,
                                                     int64_t numAct) {
  at::cuda::CUDAGuard device_guard(features.device());
  auto device = features.device().type();
  auto kernelVolume = indicePairs.size(0);
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.scalar_type(), "IndiceMaxpoolForwardKernel", [&] {
          if (device == torch::kCPU) {
            functor::SparseMaxPoolForwardFunctor<tv::CPU, scalar_t, int>
                forwardFtor;
            forwardFtor(tv::CPU(), tv::torch2tv<scalar_t>(output),
                        tv::torch2tv<const scalar_t>(features),
                        tv::torch2tv<const int>(indicePairs).subview(i), nHot);
          } else {
            functor::SparseMaxPoolForwardFunctor<tv::TorchGPU, scalar_t, int>
                forwardFtor;
            forwardFtor(tv::TorchGPU(), tv::torch2tv<scalar_t>(output),
                        tv::torch2tv<const scalar_t>(features),
                        tv::torch2tv<const int>(indicePairs).subview(i), nHot);
            TV_CHECK_CUDA_ERR();
          }
        });
  }
  return output;
}

torch::Tensor IndiceMaxpoolBackwardCUDAKernelLauncher(torch::Tensor features,
                                                      torch::Tensor outFeatures,
                                                      torch::Tensor outGrad,
                                                      torch::Tensor indicePairs,
                                                      torch::Tensor indiceNum) {
  at::cuda::CUDAGuard device_guard(features.device());
  auto device = features.device().type();
  auto numInPlanes = features.size(1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  auto kernelVolume = indicePairs.size(0);
  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0) {
      continue;
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.scalar_type(), "IndiceMaxpoolBackwardKernel", [&] {
          if (device == torch::kCPU) {
            functor::SparseMaxPoolBackwardFunctor<tv::CPU, scalar_t, int>
                backwardFtor;
            backwardFtor(tv::CPU(), tv::torch2tv<const scalar_t>(outFeatures),
                         tv::torch2tv<const scalar_t>(features),
                         tv::torch2tv<const scalar_t>(outGrad),
                         tv::torch2tv<scalar_t>(inputGrad),
                         tv::torch2tv<const int>(indicePairs).subview(i), nHot);
          } else {
            functor::SparseMaxPoolBackwardFunctor<tv::TorchGPU, scalar_t, int>
                backwardFtor;
            backwardFtor(tv::TorchGPU(),
                         tv::torch2tv<const scalar_t>(outFeatures),
                         tv::torch2tv<const scalar_t>(features),
                         tv::torch2tv<const scalar_t>(outGrad),
                         tv::torch2tv<scalar_t>(inputGrad),
                         tv::torch2tv<const int>(indicePairs).subview(i), nHot);
            TV_CHECK_CUDA_ERR();
          }
        });
  }
  return inputGrad;
}

// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>
// clang-format off
// TODO: make spconv_utils.h order agnostic
#include "../spconv_utils.h"
// clang-format on
#include <utils/spconv/spconv/mp_helper.h>
#include <utils/spconv/spconv/reordering.h>
#include <utils/spconv/tensorview/helper_launch.h>
#include <utils/spconv/tensorview/tensorview.h>

#include <chrono>
#include <limits>
#include <spconv/reordering.cuh>
#include <type_traits>
#include <utils/spconv/tensorview/helper_kernel.cuh>

#include "pytorch_cuda_helper.hpp"

namespace functor {
template <typename scalar_t, typename Index>
struct SparseGatherFunctor<tv::TorchGPU, scalar_t, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<scalar_t, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::TorchGPU &d, tv::TensorView<scalar_t> buffer,
                  tv::TensorView<const scalar_t> features,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0) return;
    int numPlanes = features.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(scalar_t);
    mp_for_each<kernel_block_t>([=, &buffer, &features, &indices,
                                 &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;
      int nHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (nHotBlock >= NumTLP) {
            gatherVecBlockKernel<scalar_t, Index, int(NumTLP), NumILP,
                                 vecload_type_t>
                <<<dim3(numPlanes / NumTLP, size / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.getStream()>>>(buffer.data(), features.data(),
                                    indices.data(), nHotBlock,
                                    numPlanes / vecloadFactor);

            TV_CHECK_CUDA_ERR();
          }
          if (size - nHotBlock > 0) {
            gatherVecKernel<scalar_t, Index, int(NumTLP), NumILP,
                            vecload_type_t>
                <<<dim3(1, numPlanes / NumTLP),
                   dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                   d.getStream()>>>(buffer.data() + nHotBlock * numPlanes,
                                    features.data(), indices.data() + nHotBlock,
                                    size - nHotBlock,
                                    numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      gatherGenericKernel<scalar_t, Index, NumTLP, NumILP>
          <<<dim3(tv::launch::DivUp(size, NumTLP),
                  tv::launch::DivUp(numPlanes, NumTLP)),
             dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
              buffer.data(), features.data(), indices.data(), size, numPlanes);
      TV_CHECK_CUDA_ERR();
    }
  }
};
template <typename scalar_t, typename Index>
struct SparseScatterAddFunctor<tv::TorchGPU, scalar_t, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<scalar_t, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::TorchGPU &d, tv::TensorView<scalar_t> outFeatures,
                  tv::TensorView<const scalar_t> buffer,
                  tv::TensorView<const Index> indices, int size, bool stable) {
    if (size <= 0) return;
    int numPlanes = outFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor =
        sizeof(vecload_type_t) / sizeof(scalar_t);  // important for half.
    mp_for_each<kernel_block_t>([=, &d, &outFeatures, &buffer, &indices,
                                 &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;
      int nHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (nHotBlock >= NumTLP) {
            scatterAddVecBlockKernel<scalar_t, Index, int(NumTLP), NumILP,
                                     vecload_type_t>
                <<<dim3(numPlanes / NumTLP, size / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.getStream()>>>(outFeatures.data(), buffer.data(),
                                    indices.data(), nHotBlock,
                                    numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }
          if (size - nHotBlock > 0) {
            scatterAddGenericKernel<scalar_t, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.getStream()>>>(
                    outFeatures.data(), buffer.data() + nHotBlock * numPlanes,
                    indices.data() + nHotBlock, size - nHotBlock, numPlanes);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });
    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      scatterAddGenericKernel<scalar_t, Index, NumTLP, NumILP>
          <<<dim3(tv::launch::DivUp(size, NumTLP),
                  tv::launch::DivUp(numPlanes, NumTLP)),
             dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
              outFeatures.data(), buffer.data(), indices.data(), size,
              numPlanes);
      TV_CHECK_CUDA_ERR();
    }
  }
};

}  // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(scalar_t, Index)                             \
  template struct functor::SparseGatherFunctor<tv::TorchGPU, scalar_t, Index>; \
  template struct functor::SparseScatterAddFunctor<tv::TorchGPU, scalar_t,     \
                                                   Index>;

#define DECLARE_GPU_SPECS(scalar_t) DECLARE_GPU_SPECS_T_INDEX(scalar_t, int);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(at::Half);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX

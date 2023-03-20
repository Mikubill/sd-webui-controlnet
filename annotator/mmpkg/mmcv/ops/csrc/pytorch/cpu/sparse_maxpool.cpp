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

#include <torch/script.h>
#include <utils/spconv/spconv/maxpool.h>

#include "pytorch_cpp_helper.hpp"

namespace functor {
template <typename scalar_t, typename Index>
struct SparseMaxPoolForwardFunctor<tv::CPU, scalar_t, Index> {
  void operator()(const tv::CPU &d, tv::TensorView<scalar_t> outFeatures,
                  tv::TensorView<const scalar_t> inFeatures,
                  tv::TensorView<const Index> indices, int size) {
    int stride = outFeatures.dim(1);
    auto outFeaturesData = outFeatures.data();
    auto inFeaturesData = inFeatures.data();
    auto indicesIn = indices.subview(0).data();
    auto indicesOut = indices.subview(1).data();
    Index idxi, idxo;
    for (int row = 0; row < size; row++) {
      idxi = indicesIn[row] * stride;
      idxo = indicesOut[row] * stride;
      for (int plane = 0; plane < stride; ++plane)
        if (outFeaturesData[idxo + plane] < inFeaturesData[idxi + plane])
          outFeaturesData[idxo + plane] = inFeaturesData[idxi + plane];
    }
  }
};

template <typename scalar_t, typename Index>
struct SparseMaxPoolBackwardFunctor<tv::CPU, scalar_t, Index> {
  void operator()(const tv::CPU &d, tv::TensorView<const scalar_t> outFeatures,
                  tv::TensorView<const scalar_t> inFeatures,
                  tv::TensorView<const scalar_t> fout,
                  tv::TensorView<scalar_t> fin,
                  tv::TensorView<const Index> indices, int size) {
    int stride = outFeatures.dim(1);
    auto outFeaturesData = outFeatures.data();
    auto inFeaturesData = inFeatures.data();
    auto foutData = fout.data();
    auto finData = fin.data();
    auto indicesIn = indices.subview(0).data();
    auto indicesOut = indices.subview(1).data();
    Index idxi, idxo;
    for (int row = 0; row < size; row++) {
      idxi = indicesIn[row] * stride;
      idxo = indicesOut[row] * stride;
      for (int plane = 0; plane < stride; ++plane)
        if (outFeaturesData[idxo + plane] == inFeaturesData[idxi + plane])
          finData[idxi + plane] += foutData[idxo + plane];
    }
  }
};

}  // namespace functor

#define DECLARE_CPU_SPECS_T_INDEX(T, Index)                                \
  template struct functor::SparseMaxPoolForwardFunctor<tv::CPU, T, Index>; \
  template struct functor::SparseMaxPoolBackwardFunctor<tv::CPU, T, Index>;

#define DECLARE_CPU_SPECS(T)         \
  DECLARE_CPU_SPECS_T_INDEX(T, int); \
  DECLARE_CPU_SPECS_T_INDEX(T, long);

DECLARE_CPU_SPECS(float);
DECLARE_CPU_SPECS(double);
DECLARE_CPU_SPECS(at::Half);

#undef DECLARE_CPU_SPECS
#undef DECLARE_CPU_SPECS_T_INDEX

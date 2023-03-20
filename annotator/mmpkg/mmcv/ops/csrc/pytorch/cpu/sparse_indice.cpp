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
#include <utils/spconv/spconv/geometry.h>
#include <utils/spconv/spconv/indice.h>

#include "pytorch_cpp_helper.hpp"

namespace functor {
template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctor<tv::CPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::CPU& d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid) {
    if (transpose)
      return getIndicePairsDeConv<Index, IndexGrid, NDim>(
          indicesIn, indicesOut, gridsOut, indicePairs, indiceNum,
          kernelSize.data(), stride.data(), padding.data(), dilation.data(),
          outSpatialShape.data());
    else
      return getIndicePairsConv<Index, IndexGrid, NDim>(
          indicesIn, indicesOut, gridsOut, indicePairs, indiceNum,
          kernelSize.data(), stride.data(), padding.data(), dilation.data(),
          outSpatialShape.data());
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateSubMIndicePairFunctor<tv::CPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::CPU& d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid) {
    return getIndicePairsSubM<Index, IndexGrid, NDim>(
        indicesIn, gridsOut, indicePairs, indiceNum, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data());
  }
};

}  // namespace functor

#define DECLARE_CPU_SPECS_INDEX_NDIM(Index, NDIM)                           \
  template struct functor::CreateConvIndicePairFunctor<tv::CPU, Index, int, \
                                                       NDIM>;               \
  template struct functor::CreateSubMIndicePairFunctor<tv::CPU, Index, int, \
                                                       NDIM>;

#define DECLARE_CPU_INDEX(Index)          \
  DECLARE_CPU_SPECS_INDEX_NDIM(Index, 1); \
  DECLARE_CPU_SPECS_INDEX_NDIM(Index, 2); \
  DECLARE_CPU_SPECS_INDEX_NDIM(Index, 3); \
  DECLARE_CPU_SPECS_INDEX_NDIM(Index, 4);

DECLARE_CPU_INDEX(int);
DECLARE_CPU_INDEX(long);

#undef DECLARE_CPU_INDEX
#undef DECLARE_CPU_SPECS_INDEX_NDIM

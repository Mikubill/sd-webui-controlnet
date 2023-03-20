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

#ifndef SPARSE_CONV_INDICE_FUNCTOR_H_
#define SPARSE_CONV_INDICE_FUNCTOR_H_
#include <utils/spconv/tensorview/tensorview.h>

namespace functor {
template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP1 {
  Index operator()(const Device& d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose);
};

template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP2 {
  Index operator()(const Device& d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid = false);
};

template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctor {
  Index operator()(const Device& d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid = false);
};

template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
struct CreateSubMIndicePairFunctor {
  Index operator()(const Device& d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid = false);
};
}  // namespace functor

#endif

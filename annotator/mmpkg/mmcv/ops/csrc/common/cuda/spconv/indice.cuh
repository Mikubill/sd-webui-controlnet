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

#ifndef INDICE_CU_H_
#define INDICE_CU_H_
#include <utils/spconv/spconv/geometry.h>
#include <utils/spconv/tensorview/tensorview.h>

#include <utils/spconv/tensorview/helper_kernel.cuh>

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void prepareIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicesOut,
    tv::TensorView<IndexGrid> gridsOut, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
      indicePairs(offset, 0, oldNum) = ix;
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(offset, 1, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void prepareDeConvIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicesOut,
    tv::TensorView<IndexGrid> gridsOut, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPosTranspose<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
      indicePairs(offset, 0, oldNum) = ix;
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(offset, 1, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void assignGridAndIndiceOutKernel(
    tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
    int numAct, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> outSpatialShape, int batchSize) {
  Index index;
  auto indicesOutPtr = indicesOut.data();
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    index = indicePairUnique[ix];
    gridsOut[index] = ix;
    index = tv::rowArrayIdxInv<Index, NDim>(
        index, indicesOutPtr + ix * (NDim + 1) + 1, outSpatialShape.data());
    indicesOut[ix * (NDim + 1)] = index % batchSize;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void assignIndicePairsKernel(
    tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
    int numActIn, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  Index index;
  int kernelVolume = indicePairs.dim(0);
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    for (int i = 0; i < kernelVolume; ++i) {
      index = indicePairs(i, 1, ix);
      if (index > -1) {
        indicePairs(i, 1, ix) = gridsOut[index];
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void prepareSubMGridKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + ix * (NDim + 1) + 1,
                                         outSpatialShape.data()) +
            spatialVolume * indicesIn(ix, 0);
    gridsOut[index] = ix;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void getSubMIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (int i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      if (gridsOut[index] > -1) {
        auto oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
        indicePairs(offset, 1, oldNum) = gridsOut[index];
        indicePairs(offset, 0, oldNum) = ix;
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void resetGridKernel(const Index *indicePairUnique,
                                tv::TensorView<IndexGrid> gridsOut,
                                int numAct) {
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    gridsOut[indicePairUnique[ix]] = -1;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void resetGridSubMKernel(
    const Index *indices, tv::TensorView<IndexGrid> gridsOut,
    const tv::SimpleVector<Index, NDim> outSpatialShape, int numAct) {
  int outSpatialShapeReg[NDim];
  for (int i = 0; i < NDim; ++i) {
    outSpatialShapeReg[i] = outSpatialShape[i];
  }
  Index spatialVolume = 1;
  auto indsPtr = indices;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index;
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    indsPtr = indices + ix * (NDim + 1);
    index = tv::rowArrayIdx<Index, NDim>(indsPtr + 1, outSpatialShapeReg);
    gridsOut[index + spatialVolume * indsPtr[0]] = -1;
  }
}

#endif

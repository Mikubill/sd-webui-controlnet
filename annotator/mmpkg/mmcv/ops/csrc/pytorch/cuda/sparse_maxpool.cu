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
#include <utils/spconv/spconv/maxpool.h>
#include <utils/spconv/spconv/mp_helper.h>
#include <utils/spconv/tensorview/helper_launch.h>
#include <utils/spconv/tensorview/tensorview.h>

#include <chrono>
#include <limits>
#include <type_traits>
#include <utils/spconv/tensorview/helper_kernel.cuh>

#include "pytorch_cuda_helper.hpp"

template <typename scalar_t, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolFwdBlockKernel(scalar_t *outFeatures,
                                      const scalar_t *inFeatures,
                                      const Index *indicesIn,
                                      const Index *indicesOut, int numHot,
                                      int numPlanes) {
  scalar_t in, out;
  int ILPStrideY[NumILP];
  Index idxo, idxi;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x; ix < numHot;
       ix += blockDim.x * gridDim.x) {
    {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        in = inFeatures[idxi];
        out = outFeatures[idxo];
        if (in > out) {
          outFeatures[idxo] = in;
        }
      }
    }
  }
}

template <typename scalar_t, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolFwdGenericBlockKernel(scalar_t *outFeatures,
                                             const scalar_t *inFeatures,
                                             const Index *indicesIn,
                                             const Index *indicesOut,
                                             int numHot, int numPlanes) {
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  scalar_t in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
      RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        in = inFeatures[RI[ilp] + iy];
        out = outFeatures[RO[ilp] + iy];
        if (in > out) {
          outFeatures[RO[ilp] + iy] = in;
        }
      }
    }
  }
}

template <typename scalar_t, typename Index, int NumTLP, int NumILP,
          typename VecType>
__global__ void maxPoolFwdVecBlockKernel(scalar_t *outFeatures,
                                         const scalar_t *inFeatures,
                                         const Index *indicesIn,
                                         const Index *indicesOut, int numHot,
                                         int numPlanes) {
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(scalar_t);
  scalar_t bufi[vecloadFactor];
  scalar_t bufo[vecloadFactor];
  Index idxi, idxo;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x * vecloadFactor; ix < numHot;
       ix += blockDim.x * gridDim.x * vecloadFactor) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];
#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        if (bufi[i] > bufo[i]) {
          bufo[i] = bufi[i];
        }
      }
      reinterpret_cast<VecType *>(outFeatures)[idxo] =
          reinterpret_cast<VecType *>(bufo)[0];
    }
  }
}

template <typename scalar_t, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolFwdGenericKernel(scalar_t *outFeatures,
                                        const scalar_t *inFeatures,
                                        const Index *indicesIn,
                                        const Index *indicesOut, int numHot,
                                        int numPlanes) {
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  scalar_t in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < numHot) {
        RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
        RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < numHot) {
          in = inFeatures[RI[ilp] + iy];
          out = outFeatures[RO[ilp] + iy];
          if (in > out) {
            outFeatures[RO[ilp] + iy] = in;
          }
        }
      }
    }
  }
}

template <typename scalar_t, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolBwdBlockKernel(const scalar_t *outFeatures,
                                      const scalar_t *inFeatures,
                                      const scalar_t *fout, scalar_t *fin,
                                      const Index *indicesIn,
                                      const Index *indicesOut, int numHot,
                                      int numPlanes) {
  scalar_t in, out;
  Index idxo, idxi;
  int ILPStrideY[NumILP];
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  fout += blockIdx.y * NumTLP;
  fin += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x; ix < numHot;
       ix += blockDim.x * gridDim.x) {
    {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
        in = inFeatures[idxi];
        out = outFeatures[idxo];
        if (in == out) {
          fin[idxi] += fout[idxo];
        }
      }
    }
  }
}

template <typename scalar_t, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolBwdGenericBlockKernel(
    const scalar_t *outFeatures, const scalar_t *inFeatures,
    const scalar_t *fout, scalar_t *fin, const Index *indicesIn,
    const Index *indicesOut, int numHot, int numPlanes) {
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  scalar_t in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
      RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        in = inFeatures[RI[ilp] + iy];
        out = outFeatures[RO[ilp] + iy];
        if (in == out) {
          fin[RI[ilp] + iy] += fout[RO[ilp] + iy];
        }
      }
    }
  }
}

template <typename scalar_t, typename Index, int NumTLP, int NumILP,
          typename VecType>
__global__ void maxPoolBwdVecBlockKernel(const scalar_t *outFeatures,
                                         const scalar_t *inFeatures,
                                         const scalar_t *fout, scalar_t *fin,
                                         const Index *indicesIn,
                                         const Index *indicesOut, int numHot,
                                         int numPlanes) {
  int ILPStrideY[NumILP];
  constexpr int vecloadFactor = sizeof(VecType) / sizeof(scalar_t);
  scalar_t bufi[vecloadFactor];
  scalar_t bufo[vecloadFactor];
  scalar_t bufdi[vecloadFactor];
  scalar_t bufdo[vecloadFactor];
  Index idxi, idxo;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideY[ilp] = threadIdx.y + ilp * blockDim.y;
  outFeatures += blockIdx.y * NumTLP;
  inFeatures += blockIdx.y * NumTLP;
  for (int ix = blockIdx.x * blockDim.x * vecloadFactor; ix < numHot;
       ix += blockDim.x * gridDim.x * vecloadFactor) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ++ilp) {
      idxi = indicesIn[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      idxo = indicesOut[ix + ILPStrideY[ilp]] * numPlanes + threadIdx.x;
      reinterpret_cast<VecType *>(bufo)[0] =
          reinterpret_cast<const VecType *>(outFeatures)[idxo];
      reinterpret_cast<VecType *>(bufi)[0] =
          reinterpret_cast<const VecType *>(inFeatures)[idxi];
      reinterpret_cast<VecType *>(bufdo)[0] =
          reinterpret_cast<const VecType *>(fout)[idxo];
      reinterpret_cast<VecType *>(bufdi)[0] =
          reinterpret_cast<VecType *>(fin)[idxi];

#pragma unroll
      for (int i = 0; i < vecloadFactor; i++) {
        if (bufi[i] == bufo[i]) {
          bufdi[i] += bufdo[i];
        }
      }
      reinterpret_cast<VecType *>(fin)[idxi] =
          reinterpret_cast<VecType *>(bufdi)[0];
    }
  }
}

template <typename scalar_t, typename Index, int NumTLP, int NumILP>
__global__ void maxPoolBwdGenericKernel(const scalar_t *outFeatures,
                                        const scalar_t *inFeatures,
                                        const scalar_t *fout, scalar_t *fin,
                                        const Index *indicesIn,
                                        const Index *indicesOut, int numHot,
                                        int numPlanes) {
  int ILPStrideX[NumILP];
  Index RI[NumILP];
  Index RO[NumILP];
  scalar_t in, out;
#pragma unroll
  for (int ilp = 0; ilp < NumILP; ilp++)
    ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
  for (int ix : tv::KernelLoopX<int, NumILP>(numHot)) {
#pragma unroll
    for (int ilp = 0; ilp < NumILP; ilp++) {
      if (ix + ILPStrideX[ilp] < numHot) {
        RI[ilp] = indicesIn[ix + ILPStrideX[ilp]] * numPlanes;
        RO[ilp] = indicesOut[ix + ILPStrideX[ilp]] * numPlanes;
      }
    }
    for (int iy : tv::KernelLoopY<int>(numPlanes)) {
#pragma unroll
      for (int ilp = 0; ilp < NumILP; ++ilp) {
        if (ix + ILPStrideX[ilp] < numHot) {
          in = inFeatures[RI[ilp] + iy];
          out = outFeatures[RO[ilp] + iy];
          if (in == out) {
            fin[RI[ilp] + iy] += fout[RO[ilp] + iy];
          }
        }
      }
    }
  }
}

namespace functor {
template <typename scalar_t, typename Index>
struct SparseMaxPoolForwardFunctor<tv::TorchGPU, scalar_t, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<scalar_t, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::TorchGPU &d, tv::TensorView<scalar_t> outFeatures,
                  tv::TensorView<const scalar_t> inFeatures,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0) return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(scalar_t);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &indices,
                                 &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            maxPoolFwdVecBlockKernel<scalar_t, Index, int(NumTLP), NumILP,
                                     vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.getStream()>>>(outFeatures.data(), inFeatures.data(),
                                    indices.subview(0).data(),
                                    indices.subview(1).data(), numHotBlock,
                                    numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            maxPoolFwdGenericKernel<scalar_t, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.getStream()>>>(outFeatures.data(), inFeatures.data(),
                                       indices.subview(0).data() + numHotBlock,
                                       indices.subview(1).data() + numHotBlock,
                                       size - numHotBlock, numPlanes);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      int numHotBlock = (size / NumTLP) * NumTLP;
      if (numHotBlock >= NumTLP) {
        maxPoolFwdGenericBlockKernel<scalar_t, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes);
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        maxPoolFwdGenericKernel<scalar_t, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
                outFeatures.data(), inFeatures.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes);
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

template <typename scalar_t, typename Index>
struct SparseMaxPoolBackwardFunctor<tv::TorchGPU, scalar_t, Index> {
  using vecload_type_t =
      std::conditional_t<std::is_same<scalar_t, at::Half>::value, int2, int4>;
  using kernel_block_t = mp_list_c<int, 64, 32, 16>;
  void operator()(const tv::TorchGPU &d,
                  tv::TensorView<const scalar_t> outFeatures,
                  tv::TensorView<const scalar_t> inFeatures,
                  tv::TensorView<const scalar_t> fout,
                  tv::TensorView<scalar_t> fin,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0) return;
    int numPlanes = inFeatures.dim(1);
    bool notFound = true;
    constexpr int vecloadFactor = sizeof(vecload_type_t) / sizeof(scalar_t);
    mp_for_each<kernel_block_t>([=, &outFeatures, &inFeatures, &fout, &fin,
                                 &indices, &notFound](auto NumTLP) {
      constexpr int NumILP = NumTLP / 4;

      int numHotBlock = (size / NumTLP) * NumTLP;
      if (notFound) {
        if (numPlanes % NumTLP == 0) {
          if (numHotBlock >= NumTLP) {
            maxPoolBwdVecBlockKernel<scalar_t, Index, int(NumTLP), NumILP,
                                     vecload_type_t>
                <<<dim3(std::min(size / NumTLP, 512), numPlanes / NumTLP),
                   dim3(NumTLP / vecloadFactor, NumTLP / NumILP), 0,
                   d.getStream()>>>(outFeatures.data(), inFeatures.data(),
                                    fout.data(), fin.data(),
                                    indices.subview(0).data(),
                                    indices.subview(1).data(), numHotBlock,
                                    numPlanes / vecloadFactor);
            TV_CHECK_CUDA_ERR();
          }

          if (size > numHotBlock) {
            maxPoolBwdGenericKernel<scalar_t, Index, int(NumTLP), NumILP>
                <<<dim3(1, numPlanes / NumTLP), dim3(NumTLP / NumILP, NumTLP),
                   0, d.getStream()>>>(outFeatures.data(), inFeatures.data(),
                                       fout.data(), fin.data(),
                                       indices.subview(0).data() + numHotBlock,
                                       indices.subview(1).data() + numHotBlock,
                                       size - numHotBlock, numPlanes);
            TV_CHECK_CUDA_ERR();
          }
          notFound = false;
        }
      }
    });

    if (notFound) {
      constexpr int NumTLP = 64;
      constexpr int NumILP = NumTLP / 4;
      int numHotBlock = (size / NumTLP) * NumTLP;
      if (numHotBlock >= NumTLP) {
        maxPoolBwdGenericBlockKernel<scalar_t, Index, NumTLP, NumILP>
            <<<dim3(size / NumTLP, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
                outFeatures.data(), inFeatures.data(), fout.data(), fin.data(),
                indices.subview(0).data(), indices.subview(1).data(),
                numHotBlock, numPlanes);
        TV_CHECK_CUDA_ERR();
      }

      if (size > numHotBlock) {
        maxPoolBwdGenericKernel<scalar_t, Index, NumTLP, NumILP>
            <<<dim3(1, tv::launch::DivUp(numPlanes, NumTLP)),
               dim3(NumTLP / NumILP, NumTLP), 0, d.getStream()>>>(
                outFeatures.data(), inFeatures.data(), fout.data(), fin.data(),
                indices.subview(0).data() + numHotBlock,
                indices.subview(1).data() + numHotBlock, size - numHotBlock,
                numPlanes);
        TV_CHECK_CUDA_ERR();
      }
    }
  }
};

}  // namespace functor

#define DECLARE_GPU_SPECS_T_INDEX(scalar_t, Index)                             \
  template struct functor::SparseMaxPoolForwardFunctor<tv::TorchGPU, scalar_t, \
                                                       Index>;                 \
  template struct functor::SparseMaxPoolBackwardFunctor<tv::TorchGPU,          \
                                                        scalar_t, Index>;

#define DECLARE_GPU_SPECS(scalar_t) DECLARE_GPU_SPECS_T_INDEX(scalar_t, int);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(at::Half);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_T_INDEX

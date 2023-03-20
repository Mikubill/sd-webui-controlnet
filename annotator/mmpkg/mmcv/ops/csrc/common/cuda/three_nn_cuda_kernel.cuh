// Copyright (c) OpenMMLab. All rights reserved
#ifndef THREE_NN_CUDA_KERNEL_CUH
#define THREE_NN_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void three_nn_forward_cuda_kernel(int b, int n, int m,
                                             const T *unknown, const T *known,
                                             T *dist2, int *__restrict__ idx) {
  // unknown: (B, N, 3)
  // known: (B, M, 3)
  // output:
  //      dist2: (B, N, 3)
  //      idx: (B, N, 3)

  int bs_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, n) {
    if (bs_idx >= b) return;

    unknown += bs_idx * n * 3 + pt_idx * 3;
    known += bs_idx * m * 3;
    dist2 += bs_idx * n * 3 + pt_idx * 3;
    idx += bs_idx * n * 3 + pt_idx * 3;

    T ux = unknown[0];
    T uy = unknown[1];
    T uz = unknown[2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
      T x = known[k * 3 + 0];
      T y = known[k * 3 + 1];
      T z = known[k * 3 + 2];
      T d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
      if (d < best1) {
        best3 = best2;
        besti3 = besti2;
        best2 = best1;
        besti2 = besti1;
        best1 = d;
        besti1 = k;
      } else if (d < best2) {
        best3 = best2;
        besti3 = besti2;
        best2 = d;
        besti2 = k;
      } else if (d < best3) {
        best3 = d;
        besti3 = k;
      }
    }
    dist2[0] = best1;
    dist2[1] = best2;
    dist2[2] = best3;
    idx[0] = besti1;
    idx[1] = besti2;
    idx[2] = besti3;
  }
}

#endif  // THREE_NN_CUDA_KERNEL_CUH

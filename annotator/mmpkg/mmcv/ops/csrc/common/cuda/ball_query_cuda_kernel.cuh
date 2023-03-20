// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu
#ifndef BALL_QUERY_CUDA_KERNEL_CUH
#define BALL_QUERY_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void ball_query_forward_cuda_kernel(int b, int n, int m,
                                               float min_radius,
                                               float max_radius, int nsample,
                                               const T* new_xyz, const T* xyz,
                                               int* idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  int bs_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, m) {
    if (bs_idx >= b) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float max_radius2 = max_radius * max_radius;
    float min_radius2 = min_radius * min_radius;
    T new_x = new_xyz[0];
    T new_y = new_xyz[1];
    T new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
      T x = xyz[k * 3 + 0];
      T y = xyz[k * 3 + 1];
      T z = xyz[k * 3 + 2];
      T d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
             (new_z - z) * (new_z - z);
      if (d2 == 0 || (d2 >= min_radius2 && d2 < max_radius2)) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[l] = k;
          }
        }
        idx[cnt] = k;
        ++cnt;
        if (cnt >= nsample) break;
      }
    }
  }
}

#endif  // BALL_QUERY_CUDA_KERNEL_CUH

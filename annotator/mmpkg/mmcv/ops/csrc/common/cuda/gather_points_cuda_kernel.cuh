// Copyright (c) OpenMMLab. All rights reserved
#ifndef GATHER_POINTS_CUDA_KERNEL_CUH
#define GATHER_POINTS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#define TOTAL_THREADS 1024

template <typename T>
__global__ void gather_points_forward_cuda_kernel(int b, int c, int n, int m,
                                                  const T *points,
                                                  const int *__restrict__ idx,
                                                  T *out) {
  // points: (B, C, N)
  // idx: (B, M)
  // output:
  //      out: (B, C, M)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, m) {
    if (bs_idx >= b || c_idx >= c) return;

    out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    points += bs_idx * c * n + c_idx * n;
    out[0] = points[idx[0]];
  }
}

template <typename T>
__global__ void gather_points_backward_cuda_kernel(int b, int c, int n, int m,
                                                   const T *grad_out,
                                                   const int *__restrict__ idx,
                                                   T *grad_points) {
  // grad_out: (B, C, M)
  // idx: (B, M)
  // output:
  //      grad_points: (B, C, N)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, m) {
    if (bs_idx >= b || c_idx >= c) return;

    grad_out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    grad_points += bs_idx * c * n + c_idx * n;

    atomicAdd(grad_points + idx[0], grad_out[0]);
  }
}

#endif  // GATHER_POINTS_CUDA_KERNEL_CUH

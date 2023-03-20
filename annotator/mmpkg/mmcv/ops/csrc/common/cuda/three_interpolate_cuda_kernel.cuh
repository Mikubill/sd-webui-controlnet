// Copyright (c) OpenMMLab. All rights reserved
#ifndef THREE_INTERPOLATE_CUDA_KERNEL_CUH
#define THREE_INTERPOLATE_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void three_interpolate_forward_cuda_kernel(
    int b, int c, int m, int n, const T *points, const int *__restrict__ idx,
    const T *weight, T *out) {
  // points: (B, C, M)
  // idx: (B, N, 3)
  // weight: (B, N, 3)
  // output:
  //      out: (B, C, N)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, n) {
    if (bs_idx >= b || c_idx >= c) return;

    weight += bs_idx * n * 3 + pt_idx * 3;
    points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;
    out += bs_idx * c * n + c_idx * n;

    out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] +
                  weight[2] * points[idx[2]];
  }
}

template <typename T>
__global__ void three_interpolate_backward_cuda_kernel(
    int b, int c, int n, int m, const T *grad_out, const int *__restrict__ idx,
    const T *weight, T *grad_points) {
  // grad_out: (B, C, N)
  // weight: (B, N, 3)
  // output:
  //      grad_points: (B, C, M)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, n) {
    if (bs_idx >= b || c_idx >= c) return;

    grad_out += bs_idx * c * n + c_idx * n + pt_idx;
    weight += bs_idx * n * 3 + pt_idx * 3;
    grad_points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;

    atomicAdd(grad_points + idx[0], grad_out[0] * weight[0]);
    atomicAdd(grad_points + idx[1], grad_out[0] * weight[1]);
    atomicAdd(grad_points + idx[2], grad_out[0] * weight[2]);
  }
}

#endif  // THREE_INTERPOLATE_CUDA_KERNEL_CUH

// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points_gpu.cu
#ifndef GROUP_POINTS_CUDA_KERNEL_CUH
#define GROUP_POINTS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void group_points_forward_cuda_kernel(int b, int c, int n,
                                                 int npoints, int nsample,
                                                 const T *points,
                                                 const int *__restrict__ idx,
                                                 T *out) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(index, npoints * nsample) {
    if (bs_idx >= b || c_idx >= c) return;

    int pt_idx = index / nsample;
    int sample_idx = index % nsample;

    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;
    int in_idx = bs_idx * c * n + c_idx * n + idx[0];
    int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample +
                  pt_idx * nsample + sample_idx;

    out[out_idx] = points[in_idx];
  }
}

template <typename T>
__global__ void group_points_backward_cuda_kernel(int b, int c, int n,
                                                  int npoints, int nsample,
                                                  const T *grad_out,
                                                  const int *__restrict__ idx,
                                                  T *grad_points) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(index, npoints * nsample) {
    int pt_idx = index / nsample;
    if (bs_idx >= b || c_idx >= c) return;

    int sample_idx = index % nsample;
    grad_out += bs_idx * c * npoints * nsample + c_idx * npoints * nsample +
                pt_idx * nsample + sample_idx;
    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;

    atomicAdd(grad_points + bs_idx * c * n + c_idx * n + idx[0], grad_out[0]);
  }
}

#endif  // GROUP_POINTS_CUDA_KERNEL_CUH

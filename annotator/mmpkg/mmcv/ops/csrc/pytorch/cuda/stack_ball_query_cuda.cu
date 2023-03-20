// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "stack_ball_query_cuda_kernel.cuh"
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

void StackBallQueryForwardCUDAKernelLauncher(float max_radius, int nsample,
                                             const Tensor new_xyz,
                                             const Tensor new_xyz_batch_cnt,
                                             const Tensor xyz,
                                             const Tensor xyz_batch_cnt,
                                             Tensor idx) {
  at::cuda::CUDAGuard device_guard(new_xyz.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  //   const float *new_xyz_ptr = new_xyz.data_ptr<float>();
  //   const float *xyz_ptr = xyz.data_ptr<float>();
  //   const int *new_xyz_batch_cnt_ptr = new_xyz_batch_cnt.data_ptr<int>();
  //   const int *xyz_batch_cnt_ptr = xyz_batch_cnt.data_ptr<int>();
  //   int *idx_ptr = idx.data_ptr<int>();

  int B = xyz_batch_cnt.size(0);
  int M = new_xyz.size(0);

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      new_xyz.scalar_type(), "stack_ball_query_forward_cuda_kernel", [&] {
        stack_ball_query_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                B, M, max_radius, nsample, new_xyz.data_ptr<scalar_t>(),
                new_xyz_batch_cnt.data_ptr<int>(), xyz.data_ptr<scalar_t>(),
                xyz_batch_cnt.data_ptr<int>(), idx.data_ptr<int>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

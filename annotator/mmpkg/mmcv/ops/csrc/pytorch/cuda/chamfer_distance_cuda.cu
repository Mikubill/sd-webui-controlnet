// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/chrdiller/pyTorchChamferDistance/blob/master/chamfer_distance/chamfer_distance.cpp
#include "chamfer_distance_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void ChamferDistanceForwardCUDAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, const Tensor dist1,
    const Tensor dist2, const Tensor idx1, const Tensor idx2) {
  int batch_size = xyz1.size(0);
  int n = xyz1.size(1);
  int m = xyz2.size(1);

  at::cuda::CUDAGuard device_guard(xyz1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      xyz1.scalar_type(), "chamfer_distance_forward_cuda_kernel", [&] {
        chamfer_distance_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK, 0, stream>>>(
                batch_size, n, xyz1.data_ptr<scalar_t>(), m,
                xyz2.data_ptr<scalar_t>(), dist1.data_ptr<scalar_t>(),
                idx1.data_ptr<int>());
      });
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      xyz1.scalar_type(), "chamfer_distance_forward_cuda_kernel", [&] {
        chamfer_distance_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK, 0, stream>>>(
                batch_size, m, xyz2.data_ptr<scalar_t>(), n,
                xyz1.data_ptr<scalar_t>(), dist2.data_ptr<scalar_t>(),
                idx2.data_ptr<int>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

void ChamferDistanceBackwardCUDAKernelLauncher(
    const Tensor xyz1, const Tensor xyz2, Tensor idx1, Tensor idx2,
    Tensor grad_dist1, Tensor grad_dist2, Tensor grad_xyz1, Tensor grad_xyz2) {
  int batch_size = xyz1.size(0);
  int n = xyz1.size(1);
  int m = xyz2.size(1);

  at::cuda::CUDAGuard device_guard(xyz1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      xyz1.scalar_type(), "chamfer_distance_backward_cuda_kernel", [&] {
        chamfer_distance_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK / 2, 0, stream>>>(
                batch_size, m, xyz1.data_ptr<scalar_t>(), n,
                xyz2.data_ptr<scalar_t>(), grad_dist1.data_ptr<scalar_t>(),
                idx1.data_ptr<int>(), grad_xyz1.data_ptr<scalar_t>(),
                grad_xyz2.data_ptr<scalar_t>());
      });
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      xyz1.scalar_type(), "chamfer_distance_backward_cuda_kernel", [&] {
        chamfer_distance_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK / 2, 0, stream>>>(
                batch_size, n, xyz2.data_ptr<scalar_t>(), m,
                xyz1.data_ptr<scalar_t>(), grad_dist2.data_ptr<scalar_t>(),
                idx2.data_ptr<int>(), grad_xyz2.data_ptr<scalar_t>(),
                grad_xyz1.data_ptr<scalar_t>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

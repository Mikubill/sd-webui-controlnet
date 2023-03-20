// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "three_interpolate_cuda_kernel.cuh"

void ThreeInterpolateForwardCUDAKernelLauncher(int b, int c, int m, int n,
                                               const Tensor points,
                                               const Tensor idx,
                                               const Tensor weight,
                                               Tensor out) {
  // points: (B, C, M)
  // idx: (B, N, 3)
  // weight: (B, N, 3)
  // output:
  //      out: (B, C, N)

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(n, THREADS_PER_BLOCK), c, b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "three_interpolate_forward_cuda_kernel", [&] {
        three_interpolate_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, m, n, points.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                weight.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void ThreeInterpolateBackwardCUDAKernelLauncher(int b, int c, int n, int m,
                                                const Tensor grad_out,
                                                const Tensor idx,
                                                const Tensor weight,
                                                Tensor grad_points) {
  // grad_out: (B, C, N)
  // weight: (B, N, 3)
  // output:
  //      grad_points: (B, C, M)

  at::cuda::CUDAGuard device_guard(grad_out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(n, THREADS_PER_BLOCK), c, b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "three_interpolate_backward_cuda_kernel", [&] {
        three_interpolate_backward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, n, m, grad_out.data_ptr<scalar_t>(), idx.data_ptr<int>(),
                weight.data_ptr<scalar_t>(), grad_points.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

#include <stdio.h>
#include <stdlib.h>

#include "gather_points_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void GatherPointsForwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                           const Tensor points,
                                           const Tensor idx, Tensor out) {
  // points: (B, C, N)
  // idx: (B, npoints)
  // output:
  //      out: (B, C, npoints)

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(npoints, THREADS_PER_BLOCK), c, b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "gather_points_forward_cuda_kernel", [&] {
        gather_points_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, n, npoints, points.data_ptr<scalar_t>(),
                idx.data_ptr<int>(), out.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void GatherPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                            const Tensor grad_out,
                                            const Tensor idx,
                                            Tensor grad_points) {
  // grad_out: (B, C, npoints)
  // idx: (B, npoints)
  // output:
  //      grad_points: (B, C, N)

  at::cuda::CUDAGuard device_guard(grad_out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(npoints, THREADS_PER_BLOCK), c, b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "gather_points_backward_cuda_kernel", [&] {
        gather_points_backward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, n, npoints, grad_out.data_ptr<scalar_t>(),
                idx.data_ptr<int>(), grad_points.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

// Modified from
// https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/paconv_lib/src/gpu
#include <stdio.h>
#include <stdlib.h>

#include "assign_score_withk_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void AssignScoreWithKForwardCUDAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& points, const Tensor& centers, const Tensor& scores,
    const Tensor& knn_idx, Tensor& output) {
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(GET_BLOCKS(B * O * N1 * K, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "assign_score_withk_forward_cuda_kernel", [&] {
        assign_score_withk_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                B, N0, N1, M, K, O, aggregate, points.data_ptr<scalar_t>(),
                centers.data_ptr<scalar_t>(), scores.data_ptr<scalar_t>(),
                knn_idx.data_ptr<int64_t>(), output.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void AssignScoreWithKBackwardCUDAKernelLauncher(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores) {
  at::cuda::CUDAGuard device_guard(grad_out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks1(GET_BLOCKS(B * M * O, THREADS_PER_BLOCK));
  dim3 threads1(THREADS_PER_BLOCK);
  dim3 blocks2(GET_BLOCKS(B * N1 * K * M, THREADS_PER_BLOCK));
  dim3 threads2(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "assign_score_withk_points_backward_cuda_kernel",
      [&] {
        assign_score_withk_points_backward_cuda_kernel<scalar_t>
            <<<blocks1, threads1, 0, stream>>>(
                B, N0, N1, M, K, O, aggregate, grad_out.data_ptr<scalar_t>(),
                scores.data_ptr<scalar_t>(), knn_idx.data_ptr<int64_t>(),
                grad_points.data_ptr<scalar_t>(),
                grad_centers.data_ptr<scalar_t>());
      });

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "assign_score_withk_scores_backward_cuda_kernel",
      [&] {
        assign_score_withk_scores_backward_cuda_kernel<scalar_t>
            <<<blocks2, threads2, 0, stream>>>(
                B, N0, N1, M, K, O, aggregate, grad_out.data_ptr<scalar_t>(),
                points.data_ptr<scalar_t>(), centers.data_ptr<scalar_t>(),
                knn_idx.data_ptr<int64_t>(), grad_scores.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

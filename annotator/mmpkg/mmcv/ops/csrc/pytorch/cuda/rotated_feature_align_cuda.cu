// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection/blob/master/mmdet/ops/fr/src/feature_refine_kernel.cu
#include "pytorch_cuda_helper.hpp"
#include "rotated_feature_align_cuda_kernel.cuh"

void RotatedFeatureAlignForwardCUDAKernelLauncher(const Tensor features,
                                                  const Tensor best_bboxes,
                                                  const float spatial_scale,
                                                  const int points,
                                                  Tensor output) {
  at::cuda::CUDAGuard device_guard(features.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int output_size = features.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "rotated_feature_align_forward_cuda_kernel",
      ([&] {
        const scalar_t* bottom_data = features.data_ptr<scalar_t>();
        const scalar_t* bboxes_data = best_bboxes.data_ptr<scalar_t>();
        scalar_t* top_data = output.data_ptr<scalar_t>();

        rotated_feature_align_forward_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, points, bottom_data, bboxes_data,
                scalar_t(spatial_scale), features.size(1), features.size(2),
                features.size(3), top_data);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void RotatedFeatureAlignBackwardCUDAKernelLauncher(const Tensor top_grad,
                                                   const Tensor best_bboxes,
                                                   const float spatial_scale,
                                                   const int points,
                                                   Tensor bottom_grad) {
  at::cuda::CUDAGuard device_guard(top_grad.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int output_size = top_grad.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "rotated_feature_align_backward_cuda_kernel",
      ([&] {
        const scalar_t* top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t* bboxes_data = best_bboxes.data_ptr<scalar_t>();
        scalar_t* bottom_diff = bottom_grad.data_ptr<scalar_t>();

        rotated_feature_align_backward_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, points, top_diff, bboxes_data,
                scalar_t(spatial_scale), top_grad.size(1), top_grad.size(2),
                top_grad.size(3), bottom_diff);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

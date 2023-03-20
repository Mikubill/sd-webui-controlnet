// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "riroi_align_rotated_cuda_kernel.cuh"

void RiROIAlignRotatedForwardCUDAKernelLauncher(
    const at::Tensor features, const at::Tensor rois, const float spatial_scale,
    const int num_samples, const bool clockwise, const int channels,
    const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, const int num_orientations,
    at::Tensor output) {
  const int output_size =
      num_rois * pooled_height * pooled_width * channels * num_orientations;
  at::cuda::CUDAGuard device_guard(features.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "riroi_align_rotated_forward_cuda_kernel", ([&] {
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        const scalar_t *rois_data = rois.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        riroi_align_rotated_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                num_samples, clockwise, channels, height, width, pooled_height,
                pooled_width, num_orientations, top_data);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
}

void RiROIAlignRotatedBackwardCUDAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int num_samples, const bool clockwise, const int channels,
    const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, const int num_orientations,
    at::Tensor bottom_grad) {
  const int output_size =
      num_rois * pooled_height * pooled_width * channels * num_orientations;
  at::cuda::CUDAGuard device_guard(top_grad.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "riroi_align_rotated_backward_cuda_kernel", ([&] {
        const scalar_t *top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t *rois_data = rois.data_ptr<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data_ptr<scalar_t>();
        riroi_align_rotated_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, top_diff, rois_data, spatial_scale, num_samples,
                clockwise, channels, height, width, pooled_height, pooled_width,
                num_orientations, bottom_diff);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

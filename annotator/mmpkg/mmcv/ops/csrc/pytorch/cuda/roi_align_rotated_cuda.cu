// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "roi_align_rotated_cuda_kernel.cuh"

void ROIAlignRotatedForwardCUDAKernelLauncher(
    const at::Tensor input, const at::Tensor rois, const float spatial_scale,
    const int sampling_ratio, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, at::Tensor output) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ROIAlignRotatedLaucherForward", ([&] {
        const scalar_t *bottom_data = input.data_ptr<scalar_t>();
        const scalar_t *rois_data = rois.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        roi_align_rotated_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sampling_ratio, aligned, clockwise, channels, height, width,
                pooled_height, pooled_width, top_data);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
}

void ROIAlignRotatedBackwardCUDAKernelLauncher(
    const at::Tensor top_grad, const at::Tensor rois, const float spatial_scale,
    const int sampling_ratio, const bool aligned, const bool clockwise,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "ROIAlignLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t *rois_data = rois.data_ptr<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data_ptr<scalar_t>();
        roi_align_rotated_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, spatial_scale, sampling_ratio,
                aligned, clockwise, channels, height, width, pooled_height,
                pooled_width, bottom_diff);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

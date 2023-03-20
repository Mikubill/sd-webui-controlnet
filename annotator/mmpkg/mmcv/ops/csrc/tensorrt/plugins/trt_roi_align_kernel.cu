// Copyright (c) OpenMMLab. All rights reserved
#include "common_cuda_helper.hpp"
#include "roi_align_cuda_kernel.cuh"

template <typename scalar_t>
void TRTRoIAlignForwardCUDAKernelLauncher(
    const scalar_t* input, const scalar_t* rois, scalar_t* output,
    scalar_t* argmax_y, scalar_t* argmax_x, int output_size, int channels,
    int height, int width, int aligned_height, int aligned_width,
    scalar_t spatial_scale, int sampling_ratio, int pool_mode, bool aligned,
    cudaStream_t stream) {
  roi_align_forward_cuda_kernel<scalar_t>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          output_size, input, rois, output, argmax_y, argmax_x, aligned_height,
          aligned_width, static_cast<scalar_t>(spatial_scale), sampling_ratio,
          pool_mode, aligned, channels, height, width);
}

void TRTRoIAlignForwardCUDAKernelLauncher_float(
    const float* input, const float* rois, float* output, float* argmax_y,
    float* argmax_x, int output_size, int channels, int height, int width,
    int aligned_height, int aligned_width, float spatial_scale,
    int sampling_ratio, int pool_mode, bool aligned, cudaStream_t stream) {
  TRTRoIAlignForwardCUDAKernelLauncher<float>(
      input, rois, output, argmax_y, argmax_x, output_size, channels, height,
      width, aligned_height, aligned_width, spatial_scale, sampling_ratio,
      pool_mode, aligned, stream);
}

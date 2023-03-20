// Copyright (c) OpenMMLab. All rights reserved
#include "deform_roi_pool_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void DeformRoIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois,
                                            Tensor offset, Tensor output,
                                            int pooled_height, int pooled_width,
                                            float spatial_scale,
                                            int sampling_ratio, float gamma) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "deform_roi_pool_forward_cuda_kernel", [&] {
        deform_roi_pool_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), offset.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), pooled_height, pooled_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio,
                static_cast<scalar_t>(gamma), channels, height, width);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void DeformRoIPoolBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "deform_roi_pool_backward_cuda_kernel", [&] {
        deform_roi_pool_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(), rois.data_ptr<scalar_t>(),
                offset.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
                grad_offset.data_ptr<scalar_t>(), pooled_height, pooled_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio,
                static_cast<scalar_t>(gamma), channels, height, width);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

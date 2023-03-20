// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cuda_helper.hpp"
#include "roi_pool_cuda_kernel.cuh"

void ROIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax, int pooled_height,
                                      int pooled_width, float spatial_scale) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_pool_forward_cuda_kernel", [&] {
        roi_pool_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                argmax.data_ptr<int>(), pooled_height, pooled_width,
                static_cast<scalar_t>(spatial_scale), channels, height, width);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void ROIPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                       Tensor argmax, Tensor grad_input,
                                       int pooled_height, int pooled_width,
                                       float spatial_scale) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "roi_pool_backward_cuda_kernel", [&] {
        roi_pool_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), argmax.data_ptr<int>(),
                grad_input.data_ptr<scalar_t>(), pooled_height, pooled_width,
                channels, height, width);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

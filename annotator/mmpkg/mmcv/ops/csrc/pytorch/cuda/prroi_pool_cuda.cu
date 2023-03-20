// Copyright (c) OpenMMLab. All rights reserved
#include "prroi_pool_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void PrROIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois,
                                        Tensor output, int pooled_height,
                                        int pooled_width, float spatial_scale) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  prroi_pool_forward_cuda_kernel<float>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          output_size, input.data_ptr<float>(), rois.data_ptr<float>(),
          output.data_ptr<float>(), pooled_height, pooled_width,
          static_cast<float>(spatial_scale), channels, height, width);

  AT_CUDA_CHECK(cudaGetLastError());
}

void PrROIPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                         Tensor grad_input, int pooled_height,
                                         int pooled_width,
                                         float spatial_scale) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  prroi_pool_backward_cuda_kernel<float>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          output_size, grad_output.data_ptr<float>(), rois.data_ptr<float>(),
          grad_input.data_ptr<float>(), pooled_height, pooled_width,
          static_cast<float>(spatial_scale), channels, height, width);

  AT_CUDA_CHECK(cudaGetLastError());
}

void PrROIPoolCoorBackwardCUDAKernelLauncher(Tensor output, Tensor grad_output,
                                             Tensor input, Tensor rois,
                                             Tensor grad_rois,
                                             int pooled_height,
                                             int pooled_width,
                                             float spatial_scale) {
  int output_size = grad_output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  prroi_pool_coor_backward_cuda_kernel<float>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          output_size, output.data_ptr<float>(), grad_output.data_ptr<float>(),
          input.data_ptr<float>(), rois.data_ptr<float>(),
          grad_rois.data_ptr<float>(), pooled_height, pooled_width,
          static_cast<float>(spatial_scale), channels, height, width);

  AT_CUDA_CHECK(cudaGetLastError());
}

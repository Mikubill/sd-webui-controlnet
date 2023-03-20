// Copyright (c) OpenMMLab. All rights reserved
#include "masked_conv2d_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void MaskedIm2colForwardCUDAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int kernel_h,
                                           const int kernel_w, const int pad_h,
                                           const int pad_w) {
  int channels = bottom_data.size(1);
  int height = bottom_data.size(2);
  int width = bottom_data.size(3);
  int mask_cnt = mask_h_idx.size(0);
  int output_size = mask_cnt * channels;

  at::cuda::CUDAGuard device_guard(bottom_data.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.scalar_type(), "MaskedIm2colLaucherForward", ([&] {
        const scalar_t *bottom_data_ = bottom_data.data_ptr<scalar_t>();
        const int64_t *mask_h_idx_ = mask_h_idx.data_ptr<int64_t>();
        const int64_t *mask_w_idx_ = mask_w_idx.data_ptr<int64_t>();
        scalar_t *top_data_ = top_data.data_ptr<scalar_t>();
        MaskedIm2colForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, bottom_data_, height, width, kernel_h, kernel_w,
                pad_h, pad_w, mask_h_idx_, mask_w_idx_, mask_cnt, top_data_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void MaskedCol2imForwardCUDAKernelLauncher(
    const Tensor bottom_data, const Tensor mask_h_idx, const Tensor mask_w_idx,
    Tensor top_data, const int height, const int width, const int channels) {
  int mask_cnt = mask_h_idx.size(0);
  int output_size = mask_cnt * channels;

  at::cuda::CUDAGuard device_guard(bottom_data.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.scalar_type(), "MaskedCol2imLaucherForward", ([&] {
        const scalar_t *bottom_data_ = bottom_data.data_ptr<scalar_t>();
        const int64_t *mask_h_idx_ = mask_h_idx.data_ptr<int64_t>();
        const int64_t *mask_w_idx_ = mask_w_idx.data_ptr<int64_t>();
        scalar_t *top_data_ = top_data.data_ptr<scalar_t>();

        MaskedCol2imForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, bottom_data_, height, width, channels, mask_h_idx_,
                mask_w_idx_, mask_cnt, top_data_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

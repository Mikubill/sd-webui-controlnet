// Copyright (c) OpenMMLab. All rights reserved
#include "modulated_deform_conv_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void modulated_deformable_im2col_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "modulated_deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        modulated_deformable_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, data_offset_, data_mask_, height_im,
            width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group, batch_size,
            channels, deformable_group, height_col, width_col, data_col_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void modulated_deformable_col2im_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im) {
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels =
      channels * kernel_h * kernel_w * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        modulated_deformable_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_offset_, data_mask_, channels,
            height_im, width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, deformable_group, height_col, width_col, grad_im_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void modulated_deformable_col2im_coord_cuda(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h *
                          kernel_w * deformable_group;
  const int channel_per_deformable_group =
      channels * kernel_h * kernel_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();
        scalar_t *grad_mask_ = grad_mask.data_ptr<scalar_t>();

        modulated_deformable_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_im_, data_offset_, data_mask_,
            channels, height_im, width_im, kernel_h, kernel_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, batch_size,
            2 * kernel_h * kernel_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset_, grad_mask_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

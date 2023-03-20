// Copyright (c) OpenMMLab. All rights reserved
#include "deform_conv_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void deformable_im2col_cuda(Tensor data_im, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor data_col) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, data_offset_, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, parallel_imgs, channels,
            deformable_group, height_col, width_col, data_col_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void deformable_col2im_cuda(Tensor data_col, Tensor data_offset,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs, const int deformable_group,
                            Tensor grad_im) {
  // todo: make sure parallel_imgs is passed in correctly
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_offset_, channels, height, width,
            ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
            dilation_w, channel_per_deformable_group, parallel_imgs,
            deformable_group, height_col, width_col, grad_im_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void deformable_col2im_coord_cuda(
    Tensor data_col, Tensor data_im, Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, Tensor grad_offset) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w *
                    deformable_group * parallel_imgs;
  int channel_per_deformable_group =
      channels * ksize_h * ksize_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();

        deformable_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_im_, data_offset_, channels, height,
            width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group, parallel_imgs,
            2 * ksize_h * ksize_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

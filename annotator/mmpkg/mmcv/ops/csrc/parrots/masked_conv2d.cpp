// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void masked_im2col_forward_impl(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w) {
  DISPATCH_DEVICE_IMPL(masked_im2col_forward_impl, im, mask_h_idx, mask_w_idx,
                       col, kernel_h, kernel_w, pad_h, pad_w);
}

void masked_col2im_forward_impl(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels) {
  DISPATCH_DEVICE_IMPL(masked_col2im_forward_impl, col, mask_h_idx, mask_w_idx,
                       im, height, width, channels);
}

void masked_im2col_forward(const Tensor im, const Tensor mask_h_idx,
                           const Tensor mask_w_idx, Tensor col,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w) {
  masked_im2col_forward_impl(im, mask_h_idx, mask_w_idx, col, kernel_h,
                             kernel_w, pad_h, pad_w);
}

void masked_col2im_forward(const Tensor col, const Tensor mask_h_idx,
                           const Tensor mask_w_idx, Tensor im, int height,
                           int width, int channels) {
  masked_col2im_forward_impl(col, mask_h_idx, mask_w_idx, im, height, width,
                             channels);
}

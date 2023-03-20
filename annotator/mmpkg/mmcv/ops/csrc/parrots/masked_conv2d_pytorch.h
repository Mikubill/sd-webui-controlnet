// Copyright (c) OpenMMLab. All rights reserved
#ifndef MASKED_CONV2D_PYTORCH_H
#define MASKED_CONV2D_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void masked_im2col_forward_cuda(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w);

void masked_col2im_forward_cuda(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels);
#endif  // MASKED_CONV2D_PYTORCH_H

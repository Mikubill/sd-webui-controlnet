// Copyright (c) OpenMMLab. All rights reserved
#ifndef CARAFE_NAIVE_PYTORCH_H
#define CARAFE_NAIVE_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void carafe_naive_forward_cuda(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor);

void carafe_naive_backward_cuda(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor);
#endif  // CARAFE_NAIVE_PYTORCH_H

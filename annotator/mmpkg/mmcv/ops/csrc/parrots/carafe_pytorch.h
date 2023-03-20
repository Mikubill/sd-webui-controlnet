// Copyright (c) OpenMMLab. All rights reserved
#ifndef CARAFE_PYTORCH_H
#define CARAFE_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void carafe_forward_cuda(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor);

void carafe_backward_cuda(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor);
#endif  // CARAFE_PYTORCH_H

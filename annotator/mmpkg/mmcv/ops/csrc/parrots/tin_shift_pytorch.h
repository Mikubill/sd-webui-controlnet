// Copyright (c) OpenMMLab. All rights reserved
#ifndef TIN_SHIFT_PYTORCH_H
#define TIN_SHIFT_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void tin_shift_forward_cuda(Tensor input, Tensor shift, Tensor output);

void tin_shift_backward_cuda(Tensor grad_output, Tensor shift,
                             Tensor grad_input);
#endif  // TIN_SHIFT_PYTORCH_H

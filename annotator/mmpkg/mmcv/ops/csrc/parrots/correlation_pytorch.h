// Copyright (c) OpenMMLab. All rights reserved
#ifndef CORRELATION_PYTORCH_H
#define CORRELATION_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void correlation_forward(Tensor input1, Tensor input2, Tensor output, int kH,
                         int kW, int patchH, int patchW, int padH, int padW,
                         int dilationH, int dilationW, int dilation_patchH,
                         int dilation_patchW, int dH, int dW);

void correlation_backward(Tensor grad_output, Tensor input1, Tensor input2,
                          Tensor grad_input1, Tensor grad_input2, int kH,
                          int kW, int patchH, int patchW, int padH, int padW,
                          int dilationH, int dilationW, int dilation_patchH,
                          int dilation_patchW, int dH, int dW);

#endif  // CORRELATION_PYTORCH_H

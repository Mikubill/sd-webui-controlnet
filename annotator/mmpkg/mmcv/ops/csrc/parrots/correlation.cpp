// Copyright (c) OpenMMLab. All rights reserved.
#include <iostream>

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void correlation_forward_impl(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW) {
  DISPATCH_DEVICE_IMPL(correlation_forward_impl, input1, input2, output, kH, kW,
                       patchH, patchW, padH, padW, dilationH, dilationW,
                       dilation_patchH, dilation_patchW, dH, dW);
}

void correlation_backward_impl(Tensor grad_output, Tensor input1, Tensor input2,
                               Tensor grad_input1, Tensor grad_input2, int kH,
                               int kW, int patchH, int patchW, int padH,
                               int padW, int dilationH, int dilationW,
                               int dilation_patchH, int dilation_patchW, int dH,
                               int dW) {
  DISPATCH_DEVICE_IMPL(correlation_backward_impl, grad_output, input1, input2,
                       grad_input1, grad_input2, kH, kW, patchH, patchW, padH,
                       padW, dilationH, dilationW, dilation_patchH,
                       dilation_patchW, dH, dW);
}

void correlation_forward(Tensor input1, Tensor input2, Tensor output, int kH,
                         int kW, int patchH, int patchW, int padH, int padW,
                         int dilationH, int dilationW, int dilation_patchH,
                         int dilation_patchW, int dH, int dW) {
  correlation_forward_impl(input1, input2, output, kH, kW, patchH, patchW, padH,
                           padW, dilationH, dilationW, dilation_patchH,
                           dilation_patchW, dH, dW);
}

void correlation_backward(Tensor grad_output, Tensor input1, Tensor input2,
                          Tensor grad_input1, Tensor grad_input2, int kH,
                          int kW, int patchH, int patchW, int padH, int padW,
                          int dilationH, int dilationW, int dilation_patchH,
                          int dilation_patchW, int dH, int dW) {
  correlation_backward_impl(grad_output, input1, input2, grad_input1,
                            grad_input2, kH, kW, patchH, patchW, padH, padW,
                            dilationH, dilationW, dilation_patchH,
                            dilation_patchW, dH, dW);
}

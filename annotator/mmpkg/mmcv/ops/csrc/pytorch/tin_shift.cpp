// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void tin_shift_forward_impl(Tensor input, Tensor shift, Tensor output) {
  DISPATCH_DEVICE_IMPL(tin_shift_forward_impl, input, shift, output);
}

void tin_shift_backward_impl(Tensor grad_output, Tensor shift,
                             Tensor grad_input) {
  DISPATCH_DEVICE_IMPL(tin_shift_backward_impl, grad_output, shift, grad_input);
}

void tin_shift_forward(Tensor input, Tensor shift, Tensor output) {
  tin_shift_forward_impl(input, shift, output);
}

void tin_shift_backward(Tensor grad_output, Tensor shift, Tensor grad_input) {
  tin_shift_backward_impl(grad_output, shift, grad_input);
}

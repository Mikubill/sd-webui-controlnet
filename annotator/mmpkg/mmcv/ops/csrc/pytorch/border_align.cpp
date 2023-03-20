// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void border_align_forward_impl(const Tensor &input, const Tensor &boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size) {
  DISPATCH_DEVICE_IMPL(border_align_forward_impl, input, boxes, output,
                       argmax_idx, pool_size);
}

void border_align_backward_impl(const Tensor &grad_output, const Tensor &boxes,
                                const Tensor &argmax_idx, Tensor grad_input,
                                const int pool_size) {
  DISPATCH_DEVICE_IMPL(border_align_backward_impl, grad_output, boxes,
                       argmax_idx, grad_input, pool_size);
}

void border_align_forward(const Tensor &input, const Tensor &boxes,
                          Tensor output, Tensor argmax_idx,
                          const int pool_size) {
  border_align_forward_impl(input, boxes, output, argmax_idx, pool_size);
}

void border_align_backward(const Tensor &grad_output, const Tensor &boxes,
                           const Tensor &argmax_idx, Tensor grad_input,
                           const int pool_size) {
  border_align_backward_impl(grad_output, boxes, argmax_idx, grad_input,
                             pool_size);
}

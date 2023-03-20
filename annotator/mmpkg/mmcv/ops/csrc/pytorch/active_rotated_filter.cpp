// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/csuhan/s2anet/blob/master/mmdet/ops/orn/src/ActiveRotatingFilter.h

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void active_rotated_filter_forward_impl(const Tensor input,
                                        const Tensor indices, Tensor output) {
  DISPATCH_DEVICE_IMPL(active_rotated_filter_forward_impl, input, indices,
                       output);
}

void active_rotated_filter_backward_impl(const Tensor grad_out,
                                         const Tensor indices, Tensor grad_in) {
  DISPATCH_DEVICE_IMPL(active_rotated_filter_backward_impl, grad_out, indices,
                       grad_in);
}

void active_rotated_filter_forward(const Tensor input, const Tensor indices,
                                   Tensor output) {
  active_rotated_filter_forward_impl(input, indices, output);
}

void active_rotated_filter_backward(const Tensor grad_out, const Tensor indices,
                                    Tensor grad_in) {
  active_rotated_filter_backward_impl(grad_out, indices, grad_in);
}

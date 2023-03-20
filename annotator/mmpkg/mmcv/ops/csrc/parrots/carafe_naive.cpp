// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void carafe_naive_forward_impl(Tensor features, Tensor masks, Tensor output,
                               int kernel_size, int group_size,
                               int scale_factor) {
  DISPATCH_DEVICE_IMPL(carafe_naive_forward_impl, features, masks, output,
                       kernel_size, group_size, scale_factor);
}

void carafe_naive_backward_impl(Tensor top_grad, Tensor features, Tensor masks,
                                Tensor bottom_grad, Tensor mask_grad,
                                int kernel_size, int group_size,
                                int scale_factor) {
  DISPATCH_DEVICE_IMPL(carafe_naive_backward_impl, top_grad, features, masks,
                       bottom_grad, mask_grad, kernel_size, group_size,
                       scale_factor);
}

void carafe_naive_forward(Tensor features, Tensor masks, Tensor output,
                          int kernel_size, int group_size, int scale_factor) {
  carafe_naive_forward_impl(features, masks, output, kernel_size, group_size,
                            scale_factor);
}

void carafe_naive_backward(Tensor top_grad, Tensor features, Tensor masks,
                           Tensor bottom_grad, Tensor mask_grad,
                           int kernel_size, int group_size, int scale_factor) {
  carafe_naive_backward_impl(top_grad, features, masks, bottom_grad, mask_grad,
                             kernel_size, group_size, scale_factor);
}

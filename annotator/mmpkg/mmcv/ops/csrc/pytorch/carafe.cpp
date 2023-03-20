// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void carafe_forward_impl(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor) {
  DISPATCH_DEVICE_IMPL(carafe_forward_impl, features, masks, rfeatures, routput,
                       rmasks, output, kernel_size, group_size, scale_factor);
}

void carafe_backward_impl(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor) {
  DISPATCH_DEVICE_IMPL(carafe_backward_impl, top_grad, rfeatures, masks,
                       rtop_grad, rbottom_grad_hs, rbottom_grad, rmask_grad,
                       bottom_grad, mask_grad, kernel_size, group_size,
                       scale_factor);
}

void carafe_forward(Tensor features, Tensor masks, Tensor rfeatures,
                    Tensor routput, Tensor rmasks, Tensor output,
                    int kernel_size, int group_size, int scale_factor) {
  carafe_forward_impl(features, masks, rfeatures, routput, rmasks, output,
                      kernel_size, group_size, scale_factor);
}

void carafe_backward(Tensor top_grad, Tensor rfeatures, Tensor masks,
                     Tensor rtop_grad, Tensor rbottom_grad_hs,
                     Tensor rbottom_grad, Tensor rmask_grad, Tensor bottom_grad,
                     Tensor mask_grad, int kernel_size, int group_size,
                     int scale_factor) {
  carafe_backward_impl(top_grad, rfeatures, masks, rtop_grad, rbottom_grad_hs,
                       rbottom_grad, rmask_grad, bottom_grad, mask_grad,
                       kernel_size, group_size, scale_factor);
}

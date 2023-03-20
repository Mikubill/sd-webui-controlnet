// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void roi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale) {
  DISPATCH_DEVICE_IMPL(roi_pool_forward_impl, input, rois, output, argmax,
                       pooled_height, pooled_width, spatial_scale);
}

void roi_pool_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale) {
  DISPATCH_DEVICE_IMPL(roi_pool_backward_impl, grad_output, rois, argmax,
                       grad_input, pooled_height, pooled_width, spatial_scale);
}

void roi_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax,
                      int pooled_height, int pooled_width,
                      float spatial_scale) {
  roi_pool_forward_impl(input, rois, output, argmax, pooled_height,
                        pooled_width, spatial_scale);
}

void roi_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax,
                       Tensor grad_input, int pooled_height, int pooled_width,
                       float spatial_scale) {
  roi_pool_backward_impl(grad_output, rois, argmax, grad_input, pooled_height,
                         pooled_width, spatial_scale);
}

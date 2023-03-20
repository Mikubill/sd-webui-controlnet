// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void deform_roi_pool_forward_impl(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  DISPATCH_DEVICE_IMPL(deform_roi_pool_forward_impl, input, rois, offset,
                       output, pooled_height, pooled_width, spatial_scale,
                       sampling_ratio, gamma);
}

void deform_roi_pool_backward_impl(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma) {
  DISPATCH_DEVICE_IMPL(deform_roi_pool_backward_impl, grad_output, input, rois,
                       offset, grad_input, grad_offset, pooled_height,
                       pooled_width, spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_forward(Tensor input, Tensor rois, Tensor offset,
                             Tensor output, int pooled_height, int pooled_width,
                             float spatial_scale, int sampling_ratio,
                             float gamma) {
  deform_roi_pool_forward_impl(input, rois, offset, output, pooled_height,
                               pooled_width, spatial_scale, sampling_ratio,
                               gamma);
}

void deform_roi_pool_backward(Tensor grad_output, Tensor input, Tensor rois,
                              Tensor offset, Tensor grad_input,
                              Tensor grad_offset, int pooled_height,
                              int pooled_width, float spatial_scale,
                              int sampling_ratio, float gamma) {
  deform_roi_pool_backward_impl(grad_output, input, rois, offset, grad_input,
                                grad_offset, pooled_height, pooled_width,
                                spatial_scale, sampling_ratio, gamma);
}

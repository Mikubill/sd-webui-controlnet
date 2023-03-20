// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void prroi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                             int pooled_height, int pooled_width,
                             float spatial_scale) {
  DISPATCH_DEVICE_IMPL(prroi_pool_forward_impl, input, rois, output,
                       pooled_height, pooled_width, spatial_scale);
}

void prroi_pool_backward_impl(Tensor grad_output, Tensor rois,
                              Tensor grad_input, int pooled_height,
                              int pooled_width, float spatial_scale) {
  DISPATCH_DEVICE_IMPL(prroi_pool_backward_impl, grad_output, rois, grad_input,
                       pooled_height, pooled_width, spatial_scale);
}

void prroi_pool_coor_backward_impl(Tensor output, Tensor grad_output,
                                   Tensor input, Tensor rois, Tensor grad_rois,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale) {
  DISPATCH_DEVICE_IMPL(prroi_pool_coor_backward_impl, output, grad_output,
                       input, rois, grad_rois, pooled_height, pooled_width,
                       spatial_scale);
}

void prroi_pool_forward(Tensor input, Tensor rois, Tensor output,
                        int pooled_height, int pooled_width,
                        float spatial_scale) {
  prroi_pool_forward_impl(input, rois, output, pooled_height, pooled_width,
                          spatial_scale);
}

void prroi_pool_backward(Tensor grad_output, Tensor rois, Tensor grad_input,
                         int pooled_height, int pooled_width,
                         float spatial_scale) {
  prroi_pool_backward_impl(grad_output, rois, grad_input, pooled_height,
                           pooled_width, spatial_scale);
}

void prroi_pool_coor_backward(Tensor output, Tensor grad_output, Tensor input,
                              Tensor rois, Tensor grad_rois, int pooled_height,
                              int pooled_width, float spatial_scale) {
  prroi_pool_coor_backward_impl(output, grad_output, input, rois, grad_rois,
                                pooled_height, pooled_width, spatial_scale);
}

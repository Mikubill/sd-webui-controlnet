// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void riroi_align_rotated_forward_impl(Tensor features, Tensor rois,
                                      Tensor output, int pooled_height,
                                      int pooled_width, float spatial_scale,
                                      int num_samples, int num_orientations,
                                      bool clockwise) {
  DISPATCH_DEVICE_IMPL(riroi_align_rotated_forward_impl, features, rois, output,
                       pooled_height, pooled_width, spatial_scale, num_samples,
                       num_orientations, clockwise);
}

void riroi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                       Tensor bottom_grad, int pooled_height,
                                       int pooled_width, float spatial_scale,
                                       int num_samples, int num_orientations,
                                       bool clockwise) {
  DISPATCH_DEVICE_IMPL(riroi_align_rotated_backward_impl, top_grad, rois,
                       bottom_grad, pooled_height, pooled_width, spatial_scale,
                       num_samples, num_orientations, clockwise);
}

void riroi_align_rotated_forward(Tensor features, Tensor rois, Tensor output,
                                 int pooled_height, int pooled_width,
                                 float spatial_scale, int num_samples,
                                 int num_orientations, bool clockwise) {
  riroi_align_rotated_forward_impl(features, rois, output, pooled_height,
                                   pooled_width, spatial_scale, num_samples,
                                   num_orientations, clockwise);
}

void riroi_align_rotated_backward(Tensor top_grad, Tensor rois,
                                  Tensor bottom_grad, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int num_samples, int num_orientations,
                                  bool clockwise) {
  riroi_align_rotated_backward_impl(top_grad, rois, bottom_grad, pooled_height,
                                    pooled_width, spatial_scale, num_samples,
                                    num_orientations, clockwise);
}

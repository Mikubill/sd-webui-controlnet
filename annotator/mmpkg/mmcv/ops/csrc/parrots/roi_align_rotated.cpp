// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void roi_align_rotated_forward_impl(Tensor features, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sample_ratio,
                                    bool aligned, bool clockwise) {
  DISPATCH_DEVICE_IMPL(roi_align_rotated_forward_impl, features, rois, output,
                       aligned_height, aligned_width, spatial_scale,
                       sample_ratio, aligned, clockwise);
}

void roi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sample_ratio, bool aligned,
                                     bool clockwise) {
  DISPATCH_DEVICE_IMPL(roi_align_rotated_backward_impl, top_grad, rois,
                       bottom_grad, aligned_height, aligned_width,
                       spatial_scale, sample_ratio, aligned, clockwise);
}

void roi_align_rotated_forward(Tensor input, Tensor rois, Tensor output,
                               int aligned_height, int aligned_width,
                               float spatial_scale, int sampling_ratio,
                               bool aligned, bool clockwise) {
  roi_align_rotated_forward_impl(input, rois, output, aligned_height,
                                 aligned_width, spatial_scale, sampling_ratio,
                                 aligned, clockwise);
}

void roi_align_rotated_backward(Tensor top_grad, Tensor rois,
                                Tensor bottom_grad, int aligned_height,
                                int aligned_width, float spatial_scale,
                                int sampling_ratio, bool aligned,
                                bool clockwise) {
  roi_align_rotated_backward_impl(top_grad, rois, bottom_grad, aligned_height,
                                  aligned_width, spatial_scale, sampling_ratio,
                                  aligned, clockwise);
}

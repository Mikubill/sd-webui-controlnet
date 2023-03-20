// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  DISPATCH_DEVICE_IMPL(roi_align_forward_impl, input, rois, output, argmax_y,
                       argmax_x, aligned_height, aligned_width, spatial_scale,
                       sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned) {
  DISPATCH_DEVICE_IMPL(roi_align_backward_impl, grad_output, rois, argmax_y,
                       argmax_x, grad_input, aligned_height, aligned_width,
                       spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned) {
  roi_align_forward_impl(input, rois, output, argmax_y, argmax_x,
                         aligned_height, aligned_width, spatial_scale,
                         sampling_ratio, pool_mode, aligned);
}

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned) {
  roi_align_backward_impl(grad_output, rois, argmax_y, argmax_x, grad_input,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
}

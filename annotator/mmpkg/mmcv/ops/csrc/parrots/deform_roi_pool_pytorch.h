// Copyright (c) OpenMMLab. All rights reserved
#ifndef DEFORM_ROI_POOL_PYTORCH_H
#define DEFORM_ROI_POOL_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void deform_roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma);

void deform_roi_pool_backward_cuda(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma);
#endif  // DEFORM_ROI_POOL_PYTORCH_H

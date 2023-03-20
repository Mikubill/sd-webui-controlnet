// Copyright (c) OpenMMLab. All rights reserved
#ifndef RIROI_ALIGN_ROTATED_PYTORCH_H
#define RIROI_ALIGN_ROTATED_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void riroi_align_rotated_forward(Tensor features, Tensor rois, Tensor output,
                                 int pooled_height, int pooled_width,
                                 float spatial_scale, int num_samples,
                                 int num_orientations, bool clockwise);

void riroi_align_rotated_backward(Tensor top_grad, Tensor rois,
                                  Tensor bottom_grad, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int num_samples, int num_orientations,
                                  bool clockwise);

#endif  // RIROI_ALIGN_ROTATED_PYTORCH_H

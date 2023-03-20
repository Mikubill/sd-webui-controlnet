// Copyright (c) OpenMMLab. All rights reserved
#ifndef ROTATED_FEATURE_ALIGN_PYTORCH_H
#define ROTATED_FEATURE_ALIGN_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void rotated_feature_align_forward(const Tensor features,
                                   const Tensor best_bboxes, Tensor output,
                                   const float spatial_scale, const int points);

void rotated_feature_align_backward(const Tensor top_grad,
                                    const Tensor best_bboxes,
                                    Tensor bottom_grad,
                                    const float spatial_scale,
                                    const int points);

#endif  // ROTATED_FEATURE_ALIGN_PYTORCH_H

// Copyright (c) OpenMMLab. All rights reserved
#ifndef FURTHEST_POINT_SAMPLE_PYTORCH_H
#define FURTHEST_POINT_SAMPLE_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void furthest_point_sampling_forward(Tensor points_tensor, Tensor temp_tensor,
                                     Tensor idx_tensor, int b, int n, int m);

void furthest_point_sampling_with_dist_forward(Tensor points_tensor,
                                               Tensor temp_tensor,
                                               Tensor idx_tensor, int b, int n,
                                               int m);
#endif  // FURTHEST_POINT_SAMPLE_PYTORCH_H

// Copyright (c) OpenMMLab. All rights reserved
#ifndef GROUP_POINTS_PYTORCH_H
#define GROUP_POINTS_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void group_points_forward(Tensor points_tensor, Tensor idx_tensor,
                          Tensor out_tensor, int b, int c, int n, int npoints,
                          int nsample);

void group_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                           Tensor grad_points_tensor, int b, int c, int n,
                           int npoints, int nsample);

#endif  // GROUP_POINTS_PYTORCH_H

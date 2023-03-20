// Copyright (c) OpenMMLab. All rights reserved
#ifndef GATHER_POINTS_PYTORCH_H
#define GATHER_POINTS_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void gather_points_forward(Tensor points_tensor, Tensor idx_tensor,
                           Tensor out_tensor, int b, int c, int n, int npoints);

void gather_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                            Tensor grad_points_tensor, int b, int c, int n,
                            int npoints);
#endif  // GATHER_POINTS_PYTORCH_H

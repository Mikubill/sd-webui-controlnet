// Copyright (c) OpenMMLab. All rights reserved
#ifndef THREE_INTERPOLATE_PYTORCH_H
#define THREE_INTERPOLATE_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void three_interpolate_forward(Tensor points_tensor, Tensor idx_tensor,
                               Tensor weight_tensor, Tensor out_tensor, int b,
                               int c, int m, int n);

void three_interpolate_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                                Tensor weight_tensor, Tensor grad_points_tensor,
                                int b, int c, int n, int m);
#endif  // THREE_INTERPOLATE_PYTORCH_H

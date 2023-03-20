// Copyright (c) OpenMMLab. All rights reserved
#ifndef KNN_PYTORCH_H
#define KNN_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void knn_forward(Tensor xyz_tensor, Tensor new_xyz_tensor, Tensor idx_tensor,
                 Tensor dist2_tensor, int b, int n, int m, int nsample);
#endif  // KNN_PYTORCH_H

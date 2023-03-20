// Copyright (c) OpenMMLab. All rights reserved
#ifndef ROIPOINT_POOL3D_PYTORCH_H
#define ROIPOINT_POOL3D_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void roipoint_pool3d_forward(Tensor xyz, Tensor boxes3d, Tensor pts_feature,
                             Tensor pooled_features, Tensor pooled_empty_flag);

#endif  // ROIPOINT_POOL3D_PYTORCH_H

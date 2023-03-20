// Copyright (c) OpenMMLab. All rights reserved
#ifndef ROIAWARE_POOL3D_PYTORCH_H
#define ROIAWARE_POOL3D_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void roiaware_pool3d_forward(Tensor rois, Tensor pts, Tensor pts_feature,
                             Tensor argmax, Tensor pts_idx_of_voxels,
                             Tensor pooled_features, int pool_method);

void roiaware_pool3d_backward(Tensor pts_idx_of_voxels, Tensor argmax,
                              Tensor grad_out, Tensor grad_in, int pool_method);

#endif  // ROIAWARE_POOL3D_PYTORCH_H

// Copyright (c) OpenMMLab. All rights reserved
#ifndef ACTIVE_CHAMFER_DISTANCE_PYTORCH_H
#define ACTIVE_CHAMFER_DISTANCE_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void chamfer_distance_forward(const Tensor xyz1, const Tensor xyz2,
                              const Tensor dist1, const Tensor dist2,
                              const Tensor idx1, const Tensor idx);

void chamfer_distance_backward(const Tensor xyz1, const Tensor xyz2,
                               Tensor idx1, Tensor idx2, Tensor graddist1,
                               Tensor graddist2, Tensor gradxyz1,
                               Tensor gradxyz2);

#endif  // ACTIVE_CHAMFER_DISTANCE_PYTORCH_H

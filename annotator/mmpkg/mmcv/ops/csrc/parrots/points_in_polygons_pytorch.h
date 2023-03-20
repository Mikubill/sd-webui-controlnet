// Copyright (c) OpenMMLab. All rights reserved
#ifndef POINTS_IN_POLYGONS_PYTORCH_H
#define POINTS_IN_POLYGONS_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void points_in_polygons_forward(Tensor points, Tensor polygons, Tensor output);

#endif  // POINTS_IN_POLYGONS_PYTORCH_H

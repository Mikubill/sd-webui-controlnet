// Copyright (c) OpenMMLab. All rights reserved
#ifndef MIN_AREA_POLYGONS_PYTORCH_H
#define MIN_AREA_POLYGONS_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void min_area_polygons(const Tensor pointsets, Tensor polygons);

#endif  // MIN_AREA_POLYGONS_PYTORCH_H

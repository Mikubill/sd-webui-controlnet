// Copyright (c) OpenMMLab. All rights reserved
#ifndef PIXEL_GROUP_PYTORCH_H
#define PIXEL_GROUP_PYTORCH_H
#include <torch/extension.h>
using namespace at;

std::vector<std::vector<float>> pixel_group(
    Tensor score, Tensor mask, Tensor embedding, Tensor kernel_label,
    Tensor kernel_contour, int kernel_region_num, float distance_threshold);

#endif  // PIXEL_GROUP_PYTORCH_H

// Copyright (c) OpenMMLab. All rights reserved
#ifndef CONTOUR_EXPAND_PYTORCH_H
#define CONTOUR_EXPAND_PYTORCH_H
#include <torch/extension.h>
using namespace at;

std::vector<std::vector<int>> contour_expand(Tensor kernel_mask,
                                             Tensor internal_kernel_label,
                                             int min_kernel_area,
                                             int kernel_num);

#endif  // CONTOUR_EXPAND_PYTORCH_H

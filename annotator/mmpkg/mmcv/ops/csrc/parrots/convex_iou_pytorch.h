// Copyright (c) OpenMMLab. All rights reserved
#ifndef CONVEX_IOU_PYTORCH_H
#define CONVEX_IOU_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void convex_iou(const Tensor pointsets, const Tensor polygons, Tensor ious);

void convex_giou(const Tensor pointsets, const Tensor polygons, Tensor output);

#endif  // RIROI_ALIGN_ROTATED_PYTORCH_H

// Copyright (c) OpenMMLab. All rights reserved
#ifndef BBOX_OVERLAPS_PYTORCH_H
#define BBOX_OVERLAPS_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void bbox_overlaps_cuda(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);

#endif  // BBOX_OVERLAPS_PYTORCH_H

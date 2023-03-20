// Copyright (c) OpenMMLab. All rights reserved
#ifndef DIFF_IOU_ROTATED_PYTORCH_H
#define DIFF_IOU_ROTATED_PYTORCH_H
#include <torch/extension.h>
using namespace at;

Tensor diff_iou_rotated_sort_vertices_forward_cuda(Tensor vertices, Tensor mask,
                                                   Tensor num_valid);

#endif  // DIFF_IOU_ROTATED_PYTORCH_H

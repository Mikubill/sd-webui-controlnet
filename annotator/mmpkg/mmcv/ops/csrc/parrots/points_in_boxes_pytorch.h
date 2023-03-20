// Copyright (c) OpenMMLab. All rights reserved
#ifndef POINTS_IN_BOXES_PYTORCH_H
#define POINTS_IN_BOXES_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void points_in_boxes_part_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                  Tensor box_idx_of_points_tensor);

void points_in_boxes_all_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                 Tensor box_idx_of_points_tensor);

void points_in_boxes_cpu_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                 Tensor pts_indices_tensor);

#endif  // POINTS_IN_BOXES_PYTORCH_H

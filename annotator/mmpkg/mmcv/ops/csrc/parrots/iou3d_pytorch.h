// Copyright (c) OpenMMLab. All rights reserved
#ifndef IOU_3D_PYTORCH_H
#define IOU_3D_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void iou3d_boxes_overlap_bev_forward(Tensor boxes_a, Tensor boxes_b,
                                     Tensor ans_overlap);

void iou3d_nms3d_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                         float nms_overlap_thresh);

void iou3d_nms3d_normal_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                                float nms_overlap_thresh);

#endif  // IOU_3D_PYTORCH_H

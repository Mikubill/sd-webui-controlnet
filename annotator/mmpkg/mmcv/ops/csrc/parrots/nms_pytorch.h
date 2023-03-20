// Copyright (c) OpenMMLab. All rights reserved
#ifndef NMS_PYTORCH_H
#define NMS_PYTORCH_H
#include <torch/extension.h>

at::Tensor nms(at::Tensor boxes, at::Tensor scores, float iou_threshold,
               int offset);

at::Tensor softnms(at::Tensor boxes, at::Tensor scores, at::Tensor dets,
                   float iou_threshold, float sigma, float min_score,
                   int method, int offset);

std::vector<std::vector<int> > nms_match(at::Tensor dets, float iou_threshold);

at::Tensor nms_rotated(const at::Tensor dets, const at::Tensor scores,
                       const at::Tensor order, const at::Tensor dets_sorted,
                       const float iou_threshold, const int multi_label);
#endif  // NMS_PYTORCH_H

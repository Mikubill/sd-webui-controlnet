// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void box_iou_quadri_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned) {
  DISPATCH_DEVICE_IMPL(box_iou_quadri_impl, boxes1, boxes2, ious, mode_flag,
                       aligned);
}

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
void box_iou_quadri(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                    const int mode_flag, const bool aligned) {
  box_iou_quadri_impl(boxes1, boxes2, ious, mode_flag, aligned);
}

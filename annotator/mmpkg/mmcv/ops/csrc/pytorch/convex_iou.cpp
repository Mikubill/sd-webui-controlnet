// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/SDL-GuoZonghao/BeyondBoundingBox/tree/main/mmdet/ops/iou/src
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void convex_iou_impl(const Tensor pointsets, const Tensor polygons,
                     Tensor ious) {
  DISPATCH_DEVICE_IMPL(convex_iou_impl, pointsets, polygons, ious);
}

void convex_iou(const Tensor pointsets, const Tensor polygons, Tensor ious) {
  convex_iou_impl(pointsets, polygons, ious);
}

void convex_giou_impl(const Tensor pointsets, const Tensor polygons,
                      Tensor output) {
  DISPATCH_DEVICE_IMPL(convex_giou_impl, pointsets, polygons, output);
}

void convex_giou(const Tensor pointsets, const Tensor polygons, Tensor output) {
  convex_giou_impl(pointsets, polygons, output);
}

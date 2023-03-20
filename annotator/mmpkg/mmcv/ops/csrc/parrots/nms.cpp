// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return DISPATCH_DEVICE_IMPL(nms_impl, boxes, scores, iou_threshold, offset);
}

Tensor softnms_impl(Tensor boxes, Tensor scores, Tensor dets,
                    float iou_threshold, float sigma, float min_score,
                    int method, int offset) {
  return DISPATCH_DEVICE_IMPL(softnms_impl, boxes, scores, dets, iou_threshold,
                              sigma, min_score, method, offset);
}

std::vector<std::vector<int> > nms_match_impl(Tensor dets,
                                              float iou_threshold) {
  return DISPATCH_DEVICE_IMPL(nms_match_impl, dets, iou_threshold);
}

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return nms_impl(boxes, scores, iou_threshold, offset);
}

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset) {
  return softnms_impl(boxes, scores, dets, iou_threshold, sigma, min_score,
                      method, offset);
}

std::vector<std::vector<int> > nms_match(Tensor dets, float iou_threshold) {
  return nms_match_impl(dets, iou_threshold);
}

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/nms_rotated/nms_rotated.h
#include "pytorch_cpp_helper.hpp"

Tensor nms_rotated_cpu(const Tensor dets, const Tensor scores,
                       const float iou_threshold);

#ifdef MMCV_WITH_CUDA
Tensor nms_rotated_cuda(const Tensor dets, const Tensor scores,
                        const Tensor order, const Tensor dets_sorted,
                        const float iou_threshold, const int multi_label);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
Tensor nms_rotated(const Tensor dets, const Tensor scores, const Tensor order,
                   const Tensor dets_sorted, const float iou_threshold,
                   const int multi_label) {
  assert(dets.device().is_cuda() == scores.device().is_cuda());
  if (dets.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    return nms_rotated_cuda(dets, scores, order, dets_sorted, iou_threshold,
                            multi_label);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return nms_rotated_cpu(dets, scores, iou_threshold);
}

// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset) {
  DISPATCH_DEVICE_IMPL(bbox_overlaps_impl, bboxes1, bboxes2, ious, mode,
                       aligned, offset);
}

void bbox_overlaps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset) {
  bbox_overlaps_impl(bboxes1, bboxes2, ious, mode, aligned, offset);
}

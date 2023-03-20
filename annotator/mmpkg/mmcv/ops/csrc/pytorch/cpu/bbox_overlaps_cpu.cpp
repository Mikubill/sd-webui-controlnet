// Copyright(c) OpenMMLab.All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

using torch::indexing::None;
using torch::indexing::Slice;

void bbox_overlaps_cpu_kernel(const Tensor boxes1, const Tensor boxes2,
                              Tensor ious, const int mode_flag,
                              const bool aligned, const int offset) {
  Tensor temp_ious;
  if (aligned) {
    Tensor lt = torch::max(boxes1.index({Slice(None), Slice({None, 2})}),
                           boxes2.index({Slice(None), Slice({None, 2})}));
    Tensor rb = torch::min(boxes1.index({Slice(None), Slice(2)}),
                           boxes2.index({Slice(None), Slice(2)}));
    Tensor wh = (rb - lt + offset).clamp(0.f, INT_MAX * 1.f);
    Tensor overlap = wh.index({Slice(None), 0}) * wh.index({Slice(None), 1});
    Tensor area1 = (boxes1.index({Slice(None), 2}) -
                    boxes1.index({Slice(None), 0}) + offset) *
                   (boxes1.index({Slice(None), 3}) -
                    boxes1.index({Slice(None), 1}) + offset);
    if (mode_flag == 0) {
      Tensor area2 = (boxes2.index({Slice(None), 2}) -
                      boxes2.index({Slice(None), 0}) + offset) *
                     (boxes2.index({Slice(None), 3}) -
                      boxes2.index({Slice(None), 1}) + offset);
      temp_ious = overlap / (area1 + area2 - overlap);
    } else {
      temp_ious = overlap / area1;
    }
  } else {
    Tensor lt = torch::max(boxes1.index({Slice(None), None, Slice({None, 2})}),
                           boxes2.index({Slice(None), Slice({None, 2})}));
    Tensor rb = torch::min(boxes1.index({Slice(None), None, Slice(2)}),
                           boxes2.index({Slice(None), Slice(2)}));
    Tensor wh = (rb - lt + offset).clamp(0.f, INT_MAX * 1.f);
    Tensor overlap = wh.index({"...", 0}) * wh.index({"...", 1});
    Tensor area1 = (boxes1.index({Slice(None), 2}) -
                    boxes1.index({Slice(None), 0}) + offset) *
                   (boxes1.index({Slice(None), 3}) -
                    boxes1.index({Slice(None), 1}) + offset);
    if (mode_flag == 0) {
      Tensor area2 = (boxes2.index({Slice(None), 2}) -
                      boxes2.index({Slice(None), 0}) + offset) *
                     (boxes2.index({Slice(None), 3}) -
                      boxes2.index({Slice(None), 1}) + offset);
      temp_ious =
          overlap / (area1.index({Slice(None), None}) + area2 - overlap);
    } else {
      temp_ious = overlap / area1.index({Slice(None), None});
    }
  }
  ious.copy_(temp_ious);
}

void bbox_overlaps_cpu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                       const int mode, const bool aligned, const int offset) {
  bbox_overlaps_cpu_kernel(boxes1, boxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);

REGISTER_DEVICE_IMPL(bbox_overlaps_impl, CPU, bbox_overlaps_cpu);

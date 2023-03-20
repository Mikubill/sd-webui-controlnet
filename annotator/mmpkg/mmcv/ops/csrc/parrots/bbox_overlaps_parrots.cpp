// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "bbox_overlaps_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
/*
 * void bbox_overlaps_cuda(const Tensor bboxes1, const Tensor bboxes2, Tensor
 * ious, const int mode, const bool aligned, const int offset);
 */
void bbox_overlaps_parrots(CudaContext& ctx, const SSElement& attr,
                           const OperatorBase::in_list_t& ins,
                           OperatorBase::out_list_t& outs) {
  int mode, offset;
  bool aligned;
  SSAttrs(attr)
      .get<int>("mode", mode)
      .get<bool>("aligned", aligned)
      .get<int>("offset", offset)
      .done();

  const auto& bboxes1 = buildATensor(ctx, ins[0]);
  const auto& bboxes2 = buildATensor(ctx, ins[1]);
  auto ious = buildATensor(ctx, outs[0]);
  bbox_overlaps_cuda(bboxes1, bboxes2, ious, mode, aligned, offset);
}

PARROTS_EXTENSION_REGISTER(bbox_overlaps)
    .attr("mode")
    .attr("aligned")
    .attr("offset")
    .input(2)
    .output(1)
    .apply(bbox_overlaps_parrots)
    .done();
#endif

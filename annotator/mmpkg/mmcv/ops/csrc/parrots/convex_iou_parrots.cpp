// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "convex_iou_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void convex_iou_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  auto pointsets = buildATensor(ctx, ins[0]);
  auto polygons = buildATensor(ctx, ins[1]);
  auto ious = buildATensor(ctx, outs[0]);
  convex_iou(pointsets, polygons, ious);
}

void convex_giou_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  auto pointsets = buildATensor(ctx, ins[0]);
  auto polygons = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  convex_giou(pointsets, polygons, output);
}

PARROTS_EXTENSION_REGISTER(convex_iou)
    .input(2)
    .output(1)
    .apply(convex_iou_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(convex_giou)
    .input(2)
    .output(1)
    .apply(convex_giou_forward_cuda_parrots)
    .done();

#endif

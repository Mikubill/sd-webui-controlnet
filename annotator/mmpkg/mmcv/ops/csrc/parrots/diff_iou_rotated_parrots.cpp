// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "diff_iou_rotated_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void diff_iou_rotated_sort_vertices_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  at::Tensor boxes, scores, dets;
  auto vertices = buildATensor(ctx, ins[0]);
  auto mask = buildATensor(ctx, ins[1]);
  auto num_valid = buildATensor(ctx, ins[2]);
  auto out =
      diff_iou_rotated_sort_vertices_forward_cuda(vertices, mask, num_valid);
  updateDArray(ctx, out, outs[0]);
}

PARROTS_EXTENSION_REGISTER(diff_iou_rotated_sort_vertices_forward)
    .input(3)
    .output(1)
    .apply(diff_iou_rotated_sort_vertices_forward_cuda_parrots)
    .done();
#endif

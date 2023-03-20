// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "points_in_boxes_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void points_in_boxes_part_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  auto boxes_tensor = buildATensor(ctx, ins[0]);
  auto pts_tensor = buildATensor(ctx, ins[1]);

  auto box_idx_of_points_tensor = buildATensor(ctx, outs[0]);

  points_in_boxes_part_forward(boxes_tensor, pts_tensor,
                               box_idx_of_points_tensor);
}

void points_in_boxes_all_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  auto boxes_tensor = buildATensor(ctx, ins[0]);
  auto pts_tensor = buildATensor(ctx, ins[1]);

  auto box_idx_of_points_tensor = buildATensor(ctx, outs[0]);

  points_in_boxes_all_forward(boxes_tensor, pts_tensor,
                              box_idx_of_points_tensor);
}

PARROTS_EXTENSION_REGISTER(points_in_boxes_part_forward)
    .input(2)
    .output(1)
    .apply(points_in_boxes_part_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(points_in_boxes_all_forward)
    .input(2)
    .output(1)
    .apply(points_in_boxes_all_forward_cuda_parrots)
    .done();
#endif

void points_in_boxes_forward_cpu_parrots(HostContext& ctx,
                                         const SSElement& attr,
                                         const OperatorBase::in_list_t& ins,
                                         OperatorBase::out_list_t& outs) {
  auto boxes_tensor = buildATensor(ctx, ins[0]);
  auto pts_tensor = buildATensor(ctx, ins[1]);

  auto pts_indices_tensor = buildATensor(ctx, outs[0]);

  points_in_boxes_cpu_forward(boxes_tensor, pts_tensor, pts_indices_tensor);
}

PARROTS_EXTENSION_REGISTER(points_in_boxes_cpu_forward)
    .input(2)
    .output(1)
    .apply(points_in_boxes_forward_cpu_parrots)
    .done();

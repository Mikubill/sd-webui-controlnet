// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "iou3d_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void iou3d_boxes_overlap_bev_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  auto boxes_a = buildATensor(ctx, ins[0]);
  auto boxes_b = buildATensor(ctx, ins[1]);

  auto ans_iou = buildATensor(ctx, outs[0]);

  iou3d_boxes_overlap_bev_forward(boxes_a, boxes_b, ans_iou);
}

void iou3d_nms3d_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  float nms_overlap_thresh;
  SSAttrs(attr).get<float>("nms_overlap_thresh", nms_overlap_thresh).done();

  auto boxes = buildATensor(ctx, ins[0]);

  auto keep = buildATensor(ctx, outs[0]);
  auto keep_num = buildATensor(ctx, outs[1]);

  iou3d_nms3d_forward(boxes, keep, keep_num, nms_overlap_thresh);
}

void iou3d_nms3d_normal_forward_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  float nms_overlap_thresh;
  SSAttrs(attr).get<float>("nms_overlap_thresh", nms_overlap_thresh).done();

  auto boxes = buildATensor(ctx, ins[0]);

  auto keep = buildATensor(ctx, outs[0]);
  auto keep_num = buildATensor(ctx, outs[1]);

  iou3d_nms3d_normal_forward(boxes, keep, keep_num, nms_overlap_thresh);
}

PARROTS_EXTENSION_REGISTER(iou3d_boxes_overlap_bev_forward)
    .input(2)
    .output(1)
    .apply(iou3d_boxes_overlap_bev_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(iou3d_nms3d_forward)
    .attr("nms_overlap_thresh")
    .input(1)
    .output(2)
    .apply(iou3d_nms3d_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(iou3d_nms3d_normal_forward)
    .attr("nms_overlap_thresh")
    .input(1)
    .output(2)
    .apply(iou3d_nms3d_normal_forward_cuda_parrots)
    .done();
#endif

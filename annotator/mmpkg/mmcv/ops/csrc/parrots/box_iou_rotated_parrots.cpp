// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "box_iou_rotated_pytorch.h"

using namespace parrots;

/*
 * void box_iou_rotated_cpu(const Tensor boxes1, const Tensor boxes2, Tensor
 * ious, const int mode_flag, const bool aligned);
 */
void box_iou_rotated_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                 const OperatorBase::in_list_t& ins,
                                 OperatorBase::out_list_t& outs) {
  bool aligned;
  int mode_flag;
  SSAttrs(attr)
      .get<bool>("aligned", aligned)
      .get<int>("mode_flag", mode_flag)
      .done();

  const auto& boxes1 = buildATensor(ctx, ins[0]);
  const auto& boxes2 = buildATensor(ctx, ins[1]);
  auto ious = buildATensor(ctx, outs[0]);
  box_iou_rotated_cpu(boxes1, boxes2, ious, mode_flag, aligned);
}

#ifdef MMCV_WITH_CUDA
/*
 * void box_iou_rotated_cuda(const Tensor boxes1, const Tensor boxes2, Tensor
 * ious, const int mode_flag, const bool aligned);
 */
void box_iou_rotated_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                  const OperatorBase::in_list_t& ins,
                                  OperatorBase::out_list_t& outs) {
  bool aligned;
  int mode_flag;
  SSAttrs(attr)
      .get<bool>("aligned", aligned)
      .get<int>("mode_flag", mode_flag)
      .done();

  const auto& boxes1 = buildATensor(ctx, ins[0]);
  const auto& boxes2 = buildATensor(ctx, ins[1]);
  auto ious = buildATensor(ctx, outs[0]);
  box_iou_rotated_cuda(boxes1, boxes2, ious, mode_flag, aligned);
}
#endif

PARROTS_EXTENSION_REGISTER(box_iou_rotated)
    .attr("aligned")
    .attr("mode_flag")
    .input(2)
    .output(1)
    .apply(box_iou_rotated_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(box_iou_rotated_cuda_parrots)
#endif
    .done();

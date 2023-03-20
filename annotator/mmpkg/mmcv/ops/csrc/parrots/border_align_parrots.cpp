// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "border_align_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void border_align_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
  int pool_size;
  SSAttrs(attr).get<int>("pool_size", pool_size).done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& boxes = buildATensor(ctx, ins[1]);

  auto output = buildATensor(ctx, outs[0]);
  auto argmax_idx = buildATensor(ctx, outs[1]);
  border_align_forward_cuda(input, boxes, output, argmax_idx, pool_size);
}

void border_align_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  int pool_size;
  SSAttrs(attr).get<int>("pool_size", pool_size).done();

  const auto& top_grad = buildATensor(ctx, ins[0]);
  const auto& boxes = buildATensor(ctx, ins[1]);
  const auto& argmax_idx = buildATensor(ctx, ins[2]);

  auto bottom_grad = buildATensor(ctx, outs[0]);
  border_align_backward_cuda(top_grad, boxes, argmax_idx, bottom_grad,
                             pool_size);
}

PARROTS_EXTENSION_REGISTER(border_align_forward)
    .attr("pool_size")
    .input(2)
    .output(2)
    .apply(border_align_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(border_align_backward)
    .attr("pool_size")
    .input(3)
    .output(1)
    .apply(border_align_backward_cuda_parrots)
    .done();
#endif

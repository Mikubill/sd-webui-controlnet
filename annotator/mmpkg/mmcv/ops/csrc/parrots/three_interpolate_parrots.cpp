// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "three_interpolate_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void three_interpolate_forward_cuda_parrots(CudaContext& ctx,
                                            const SSElement& attr,
                                            const OperatorBase::in_list_t& ins,
                                            OperatorBase::out_list_t& outs) {
  int b, c, m, n;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("c", c)
      .get<int>("m", m)
      .get<int>("n", n)
      .done();

  auto points_tensor = buildATensor(ctx, ins[0]);
  auto idx_tensor = buildATensor(ctx, ins[1]);
  auto weight_tensor = buildATensor(ctx, ins[2]);

  auto out_tensor = buildATensor(ctx, outs[0]);

  three_interpolate_forward(points_tensor, idx_tensor, weight_tensor,
                            out_tensor, b, c, m, n);
}

void three_interpolate_backward_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  int b, c, n, m;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("c", c)
      .get<int>("n", n)
      .get<int>("m", m)
      .done();

  auto grad_out_tensor = buildATensor(ctx, ins[0]);
  auto idx_tensor = buildATensor(ctx, ins[1]);
  auto weight_tensor = buildATensor(ctx, ins[2]);

  auto grad_points_tensor = buildATensor(ctx, outs[0]);

  three_interpolate_backward(grad_out_tensor, idx_tensor, weight_tensor,
                             grad_points_tensor, b, c, n, m);
}

PARROTS_EXTENSION_REGISTER(three_interpolate_forward)
    .attr("b")
    .attr("c")
    .attr("m")
    .attr("n")
    .input(3)
    .output(1)
    .apply(three_interpolate_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(three_interpolate_backward)
    .attr("b")
    .attr("c")
    .attr("n")
    .attr("m")
    .input(3)
    .output(1)
    .apply(three_interpolate_backward_cuda_parrots)
    .done();
#endif

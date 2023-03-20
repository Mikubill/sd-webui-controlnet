// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "group_points_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void group_points_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
  int b, c, n, npoints, nsample;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("c", c)
      .get<int>("n", n)
      .get<int>("npoints", npoints)
      .get<int>("nsample", nsample)
      .done();
  auto points_tensor = buildATensor(ctx, ins[0]);
  auto idx_tensor = buildATensor(ctx, ins[1]);

  auto out_tensor = buildATensor(ctx, outs[0]);

  group_points_forward(points_tensor, idx_tensor, out_tensor, b, c, n, npoints,
                       nsample);
}

void group_points_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  int b, c, n, npoints, nsample;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("c", c)
      .get<int>("n", n)
      .get<int>("npoints", npoints)
      .get<int>("nsample", nsample)
      .done();
  auto grad_out_tensor = buildATensor(ctx, ins[0]);
  auto idx_tensor = buildATensor(ctx, ins[1]);

  auto grad_points_tensor = buildATensor(ctx, outs[0]);

  group_points_backward(grad_out_tensor, idx_tensor, grad_points_tensor, b, c,
                        n, npoints, nsample);
}

PARROTS_EXTENSION_REGISTER(group_points_forward)
    .attr("b")
    .attr("c")
    .attr("n")
    .attr("npoints")
    .attr("nsample")
    .input(2)
    .output(1)
    .apply(group_points_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(group_points_backward)
    .attr("b")
    .attr("c")
    .attr("n")
    .attr("npoints")
    .attr("nsample")
    .input(2)
    .output(1)
    .apply(group_points_backward_cuda_parrots)
    .done();
#endif

// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "ball_query_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void ball_query_parrots(CudaContext& ctx, const SSElement& attr,
                        const OperatorBase::in_list_t& ins,
                        OperatorBase::out_list_t& outs) {
  int b, n, m, nsample;
  float min_radius, max_radius;
  SSAttrs(attr)
      .get<int>("b", b)
      .get<int>("n", n)
      .get<int>("m", m)
      .get<int>("nsample", nsample)
      .get<float>("min_radius", min_radius)
      .get<float>("max_radius", max_radius)
      .done();

  const auto& center_xyz = buildATensor(ctx, ins[0]);
  const auto& xyz = buildATensor(ctx, ins[1]);
  auto idx = buildATensor(ctx, outs[0]);
  ball_query_forward(center_xyz, xyz, idx, b, n, m, min_radius, max_radius,
                     nsample);
}

PARROTS_EXTENSION_REGISTER(ball_query_forward)
    .attr("b")
    .attr("n")
    .attr("m")
    .attr("nsample")
    .attr("min_radius")
    .attr("max_radius")
    .input(2)
    .output(1)
    .apply(ball_query_parrots)
    .done();
#endif

// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "furthest_point_sample_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void furthest_point_sample_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int b, n, m;
  SSAttrs(attr).get<int>("b", b).get<int>("n", n).get<int>("m", m).done();

  auto points_tensor = buildATensor(ctx, ins[0]);
  auto temp_tensor = buildATensor(ctx, ins[1]);

  auto idx_tensor = buildATensor(ctx, outs[0]);

  furthest_point_sampling_forward(points_tensor, temp_tensor, idx_tensor, b, n,
                                  m);
}

void furthest_point_sampling_with_dist_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int b, n, m;
  SSAttrs(attr).get<int>("b", b).get<int>("n", n).get<int>("m", m).done();

  auto points_tensor = buildATensor(ctx, ins[0]);
  auto temp_tensor = buildATensor(ctx, ins[1]);

  auto idx_tensor = buildATensor(ctx, outs[0]);

  furthest_point_sampling_with_dist_forward(points_tensor, temp_tensor,
                                            idx_tensor, b, n, m);
}
PARROTS_EXTENSION_REGISTER(furthest_point_sampling_forward)
    .attr("b")
    .attr("n")
    .attr("m")
    .input(2)
    .output(1)
    .apply(furthest_point_sample_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(furthest_point_sampling_with_dist_forward)
    .attr("b")
    .attr("n")
    .attr("m")
    .input(2)
    .output(1)
    .apply(furthest_point_sampling_with_dist_forward_cuda_parrots)
    .done();
#endif

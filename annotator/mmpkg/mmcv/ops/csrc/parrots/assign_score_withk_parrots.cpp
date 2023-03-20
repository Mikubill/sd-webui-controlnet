// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "assign_score_withk_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void assign_score_withk_forward_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  int B, N0, N1, M, K, O, aggregate;
  SSAttrs(attr)
      .get<int>("B", B)
      .get<int>("N0", N0)
      .get<int>("N1", N1)
      .get<int>("M", M)
      .get<int>("K", K)
      .get<int>("O", O)
      .get<int>("aggregate", aggregate)
      .done();

  const auto& points = buildATensor(ctx, ins[0]);
  const auto& centers = buildATensor(ctx, ins[1]);
  const auto& scores = buildATensor(ctx, ins[2]);
  const auto& knn_idx = buildATensor(ctx, ins[3]);

  auto output = buildATensor(ctx, outs[0]);
  assign_score_withk_forward(points, centers, scores, knn_idx, output, B, N0,
                             N1, M, K, O, aggregate);
}

void assign_score_withk_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int B, N0, N1, M, K, O, aggregate;
  SSAttrs(attr)
      .get<int>("B", B)
      .get<int>("N0", N0)
      .get<int>("N1", N1)
      .get<int>("M", M)
      .get<int>("K", K)
      .get<int>("O", O)
      .get<int>("aggregate", aggregate)
      .done();

  const auto& grad_out = buildATensor(ctx, ins[0]);
  const auto& points = buildATensor(ctx, ins[1]);
  const auto& centers = buildATensor(ctx, ins[2]);
  const auto& scores = buildATensor(ctx, ins[3]);
  const auto& knn_idx = buildATensor(ctx, ins[4]);

  auto grad_points = buildATensor(ctx, outs[0]);
  auto grad_centers = buildATensor(ctx, outs[1]);
  auto grad_scores = buildATensor(ctx, outs[2]);
  assign_score_withk_backward(grad_out, points, centers, scores, knn_idx,
                              grad_points, grad_centers, grad_scores, B, N0, N1,
                              M, K, O, aggregate);
}

PARROTS_EXTENSION_REGISTER(assign_score_withk_forward)
    .attr("B")
    .attr("N0")
    .attr("N1")
    .attr("M")
    .attr("K")
    .attr("O")
    .attr("aggregate")
    .input(4)
    .output(1)
    .apply(assign_score_withk_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(assign_score_withk_backward)
    .attr("B")
    .attr("N0")
    .attr("N1")
    .attr("M")
    .attr("K")
    .attr("O")
    .attr("aggregate")
    .input(5)
    .output(3)
    .apply(assign_score_withk_backward_cuda_parrots)
    .done();
#endif

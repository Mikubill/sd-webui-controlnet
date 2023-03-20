// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "rotated_feature_align_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void rotated_feature_align_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float spatial_scale;
  int points;
  SSAttrs(attr)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("points", points)
      .done();

  auto features = buildATensor(ctx, ins[0]);
  auto best_bboxes = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  rotated_feature_align_forward(features, best_bboxes, output, spatial_scale,
                                points);
}

void rotated_feature_align_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float spatial_scale;
  int points;
  SSAttrs(attr)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("points", points)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto best_bboxes = buildATensor(ctx, ins[1]);
  auto grad_input = buildATensor(ctx, outs[0]);
  rotated_feature_align_backward(grad_output, best_bboxes, grad_input,
                                 spatial_scale, points);
}
#endif

void rotated_feature_align_forward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float spatial_scale;
  int points;
  SSAttrs(attr)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("points", points)
      .done();

  auto features = buildATensor(ctx, ins[0]);
  auto best_bboxes = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  rotated_feature_align_forward(features, best_bboxes, output, spatial_scale,
                                points);
}

void rotated_feature_align_backward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float spatial_scale;
  int points;
  SSAttrs(attr)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("points", points)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto best_bboxes = buildATensor(ctx, ins[1]);
  auto grad_input = buildATensor(ctx, outs[0]);
  rotated_feature_align_backward(grad_output, best_bboxes, grad_input,
                                 spatial_scale, points);
}

PARROTS_EXTENSION_REGISTER(rotated_feature_align_forward)
    .attr("spatial_scale")
    .attr("points")
    .input(2)
    .output(1)
    .apply(rotated_feature_align_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(rotated_feature_align_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(rotated_feature_align_backward)
    .attr("spatial_scale")
    .attr("points")
    .input(2)
    .output(1)
    .apply(rotated_feature_align_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(rotated_feature_align_backward_cuda_parrots)
#endif
    .done();

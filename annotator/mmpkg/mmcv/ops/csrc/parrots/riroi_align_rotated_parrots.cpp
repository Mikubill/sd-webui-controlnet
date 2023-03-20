// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "riroi_align_rotated_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void riroi_align_rotated_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sample_num;
  int num_orientations;
  bool clockwise;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("num_samples", sample_num)
      .get<int>("num_orientations", num_orientations)
      .get<bool>("clockwise", clockwise)
      .done();

  auto input = buildATensor(ctx, ins[0]);
  auto rois = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  riroi_align_rotated_forward(input, rois, output, pooled_height, pooled_width,
                              spatial_scale, sample_num, num_orientations,
                              clockwise);
}

void riroi_align_rotated_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sample_num;
  int num_orientations;
  bool clockwise;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("num_samples", sample_num)
      .get<int>("num_orientations", num_orientations)
      .get<bool>("clockwise", clockwise)
      .done();

  auto grad_output = buildATensor(ctx, ins[0]);
  auto rois = buildATensor(ctx, ins[1]);
  auto grad_input = buildATensor(ctx, outs[0]);
  riroi_align_rotated_backward(grad_output, rois, grad_input, pooled_height,
                               pooled_width, spatial_scale, sample_num,
                               num_orientations, clockwise);
}

PARROTS_EXTENSION_REGISTER(riroi_align_rotated_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .attr("num_samples")
    .attr("num_orientations")
    .attr("clockwise")
    .input(2)
    .output(1)
    .apply(riroi_align_rotated_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(riroi_align_rotated_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .attr("num_samples")
    .attr("num_orientations")
    .attr("clockwise")
    .input(2)
    .output(1)
    .apply(riroi_align_rotated_backward_cuda_parrots)
    .done();

#endif

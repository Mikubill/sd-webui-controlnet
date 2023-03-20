// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "roi_align_rotated_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void roi_align_rotated_forward_cuda_parrots(CudaContext& ctx,
                                            const SSElement& attr,
                                            const OperatorBase::in_list_t& ins,
                                            OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sampling_ratio;
  bool aligned;
  bool clockwise;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<bool>("aligned", aligned)
      .get<bool>("clockwise", clockwise)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  roi_align_rotated_forward_cuda(input, rois, output, pooled_height,
                                 pooled_width, spatial_scale, sampling_ratio,
                                 aligned, clockwise);
}

void roi_align_rotated_backward_cuda_parrots(CudaContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sampling_ratio;
  bool aligned;
  bool clockwise;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<bool>("aligned", aligned)
      .get<bool>("clockwise", clockwise)
      .done();

  const auto& grad_output = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  auto grad_input = buildATensor(ctx, outs[0]);
  roi_align_rotated_backward_cuda(grad_output, rois, grad_input, pooled_height,
                                  pooled_width, spatial_scale, sampling_ratio,
                                  aligned, clockwise);
}
#endif

void roi_align_rotated_forward_cpu_parrots(HostContext& ctx,
                                           const SSElement& attr,
                                           const OperatorBase::in_list_t& ins,
                                           OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sampling_ratio;
  bool aligned;
  bool clockwise;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<bool>("aligned", aligned)
      .get<bool>("clockwise", clockwise)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  roi_align_rotated_forward_cpu(input, rois, output, pooled_height,
                                pooled_width, spatial_scale, sampling_ratio,
                                aligned, clockwise);
}

void roi_align_rotated_backward_cpu_parrots(HostContext& ctx,
                                            const SSElement& attr,
                                            const OperatorBase::in_list_t& ins,
                                            OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sampling_ratio;
  bool aligned;
  bool clockwise;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<bool>("aligned", aligned)
      .get<bool>("clockwise", clockwise)
      .done();

  const auto& grad_output = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  auto grad_input = buildATensor(ctx, outs[0]);
  roi_align_rotated_backward_cpu(grad_output, rois, grad_input, pooled_height,
                                 pooled_width, spatial_scale, sampling_ratio,
                                 aligned, clockwise);
}

PARROTS_EXTENSION_REGISTER(roi_align_rotated_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("aligned")
    .attr("clockwise")
    .input(2)
    .output(1)
    .apply(roi_align_rotated_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(roi_align_rotated_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(roi_align_rotated_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("aligned")
    .attr("clockwise")
    .input(2)
    .output(1)
    .apply(roi_align_rotated_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(roi_align_rotated_backward_cuda_parrots)
#endif
    .done();

// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "roi_align_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void roi_align_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  int aligned_height;
  int aligned_width;
  float spatial_scale;
  int sampling_ratio;
  int pool_mode;
  bool aligned;
  SSAttrs(attr)
      .get<int>("aligned_height", aligned_height)
      .get<int>("aligned_width", aligned_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<int>("pool_mode", pool_mode)
      .get<bool>("aligned", aligned)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  auto argmax_y = buildATensor(ctx, outs[1]);
  auto argmax_x = buildATensor(ctx, outs[2]);
  roi_align_forward_cuda(input, rois, output, argmax_y, argmax_x,
                         aligned_height, aligned_width, spatial_scale,
                         sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  int aligned_height;
  int aligned_width;
  float spatial_scale;
  int sampling_ratio;
  int pool_mode;
  bool aligned;
  SSAttrs(attr)
      .get<int>("aligned_height", aligned_height)
      .get<int>("aligned_width", aligned_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<int>("pool_mode", pool_mode)
      .get<bool>("aligned", aligned)
      .done();

  const auto& grad_output = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  const auto& argmax_y = buildATensor(ctx, ins[2]);
  const auto& argmax_x = buildATensor(ctx, ins[3]);
  auto grad_input = buildATensor(ctx, outs[0]);
  roi_align_backward_cuda(grad_output, rois, argmax_y, argmax_x, grad_input,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
}
#endif

void roi_align_forward_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                   const OperatorBase::in_list_t& ins,
                                   OperatorBase::out_list_t& outs) {
  int aligned_height;
  int aligned_width;
  float spatial_scale;
  int sampling_ratio;
  int pool_mode;
  bool aligned;
  SSAttrs(attr)
      .get<int>("aligned_height", aligned_height)
      .get<int>("aligned_width", aligned_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<int>("pool_mode", pool_mode)
      .get<bool>("aligned", aligned)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  auto argmax_y = buildATensor(ctx, outs[1]);
  auto argmax_x = buildATensor(ctx, outs[2]);
  roi_align_forward_cpu(input, rois, output, argmax_y, argmax_x, aligned_height,
                        aligned_width, spatial_scale, sampling_ratio, pool_mode,
                        aligned);
}

void roi_align_backward_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  int aligned_height;
  int aligned_width;
  float spatial_scale;
  int sampling_ratio;
  int pool_mode;
  bool aligned;
  SSAttrs(attr)
      .get<int>("aligned_height", aligned_height)
      .get<int>("aligned_width", aligned_width)
      .get<float>("spatial_scale", spatial_scale)
      .get<int>("sampling_ratio", sampling_ratio)
      .get<int>("pool_mode", pool_mode)
      .get<bool>("aligned", aligned)
      .done();

  const auto& grad_output = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  const auto& argmax_y = buildATensor(ctx, ins[2]);
  const auto& argmax_x = buildATensor(ctx, ins[3]);
  auto grad_input = buildATensor(ctx, outs[0]);
  roi_align_backward_cpu(grad_output, rois, argmax_y, argmax_x, grad_input,
                         aligned_height, aligned_width, spatial_scale,
                         sampling_ratio, pool_mode, aligned);
}

PARROTS_EXTENSION_REGISTER(roi_align_forward)
    .attr("aligned_height")
    .attr("aligned_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("pool_mode")
    .attr("aligned")
    .input(2)
    .output(3)
    .apply(roi_align_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(roi_align_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(roi_align_backward)
    .attr("aligned_height")
    .attr("aligned_width")
    .attr("spatial_scale")
    .attr("sampling_ratio")
    .attr("pool_mode")
    .attr("aligned")
    .input(4)
    .output(1)
    .apply(roi_align_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(roi_align_backward_cuda_parrots)
#endif
    .done();

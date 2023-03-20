// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "roi_pool_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void roi_pool_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                   const OperatorBase::in_list_t& ins,
                                   OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  auto argmax = buildATensor(ctx, outs[1]);
  roi_pool_forward_cuda(input, rois, output, argmax, pooled_height,
                        pooled_width, spatial_scale);
}

void roi_pool_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  SSAttrs(attr)
      .get<int>("pooled_height", pooled_height)
      .get<int>("pooled_width", pooled_width)
      .get<float>("spatial_scale", spatial_scale)
      .done();

  const auto& grad_output = buildATensor(ctx, ins[0]);
  const auto& rois = buildATensor(ctx, ins[1]);
  const auto& argmax = buildATensor(ctx, ins[2]);
  auto grad_input = buildATensor(ctx, outs[0]);
  roi_pool_backward_cuda(grad_output, rois, argmax, grad_input, pooled_height,
                         pooled_width, spatial_scale);
}

PARROTS_EXTENSION_REGISTER(roi_pool_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(2)
    .output(2)
    .apply(roi_pool_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(roi_pool_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(3)
    .output(1)
    .apply(roi_pool_backward_cuda_parrots)
    .done();
#endif

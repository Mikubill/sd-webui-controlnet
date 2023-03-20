// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "prroi_pool_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void prroi_pool_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
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
  prroi_pool_forward(input, rois, output, pooled_height, pooled_width,
                     spatial_scale);
}

void prroi_pool_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
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
  auto grad_input = buildATensor(ctx, outs[0]);
  prroi_pool_backward(grad_output, rois, grad_input, pooled_height,
                      pooled_width, spatial_scale);
}

void prroi_pool_coor_backward_cuda_parrots(CudaContext& ctx,
                                           const SSElement& attr,
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

  const auto& output = buildATensor(ctx, ins[0]);
  const auto& grad_output = buildATensor(ctx, ins[1]);
  const auto& input = buildATensor(ctx, ins[2]);
  const auto& rois = buildATensor(ctx, ins[3]);
  auto grad_rois = buildATensor(ctx, outs[0]);
  prroi_pool_coor_backward(output, grad_output, input, rois, grad_rois,
                           pooled_height, pooled_width, spatial_scale);
}

PARROTS_EXTENSION_REGISTER(prroi_pool_forward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(2)
    .output(1)
    .apply(prroi_pool_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(prroi_pool_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(2)
    .output(1)
    .apply(prroi_pool_backward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(prroi_pool_coor_backward)
    .attr("pooled_height")
    .attr("pooled_width")
    .attr("spatial_scale")
    .input(4)
    .output(1)
    .apply(prroi_pool_coor_backward_cuda_parrots)
    .done();
#endif

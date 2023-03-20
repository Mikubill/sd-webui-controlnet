// Copyright (c) OpenMMLab. All rights reserved
#include <torch/extension.h>

#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
using namespace at;
using namespace parrots;

Tensor ms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight, const int im2col_step);

void ms_deform_attn_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &attn_weight,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             Tensor &grad_attn_weight, const int im2col_step);

void ms_deform_attn_forward_parrots(CudaContext &ctx, const SSElement &attr,
                                    const OperatorBase::in_list_t &ins,
                                    OperatorBase::out_list_t &outs) {
  int im2col_step;
  SSAttrs(attr).get<int>("im2col_step", im2col_step).done();
  const auto &value = buildATensor(ctx, ins[0]);
  const auto &spatial_shapes = buildATensor(ctx, ins[1]);
  const auto &level_start_index = buildATensor(ctx, ins[2]);
  const auto &sampling_loc = buildATensor(ctx, ins[3]);
  const auto &attn_weight = buildATensor(ctx, ins[4]);
  auto out = ms_deform_attn_forward(value, spatial_shapes, level_start_index,
                                    sampling_loc, attn_weight, im2col_step);
  updateDArray(ctx, out, outs[0]);
}

void ms_deform_attn_backward_parrots(CudaContext &ctx, const SSElement &attr,
                                     const OperatorBase::in_list_t &ins,
                                     OperatorBase::out_list_t &outs) {
  int im2col_step;
  SSAttrs(attr).get<int>("im2col_step", im2col_step).done();
  const auto &value = buildATensor(ctx, ins[0]);
  const auto &spatial_shapes = buildATensor(ctx, ins[1]);
  const auto &level_start_index = buildATensor(ctx, ins[2]);
  const auto &sampling_loc = buildATensor(ctx, ins[3]);
  const auto &attn_weight = buildATensor(ctx, ins[4]);
  const auto &grad_output = buildATensor(ctx, ins[5]);
  auto grad_value = buildATensor(ctx, outs[0]);
  auto grad_sampling_loc = buildATensor(ctx, outs[1]);
  auto grad_attn_weight = buildATensor(ctx, outs[2]);
  ms_deform_attn_backward(value, spatial_shapes, level_start_index,
                          sampling_loc, attn_weight, grad_output, grad_value,
                          grad_sampling_loc, grad_attn_weight, im2col_step);
}

PARROTS_EXTENSION_REGISTER(ms_deform_attn_forward)
    .attr("im2col_step")
    .input(5)
    .output(1)
    .apply(ms_deform_attn_forward_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(ms_deform_attn_backward)
    .attr("im2col_step")
    .input(6)
    .output(3)
    .apply(ms_deform_attn_backward_parrots)
    .done();

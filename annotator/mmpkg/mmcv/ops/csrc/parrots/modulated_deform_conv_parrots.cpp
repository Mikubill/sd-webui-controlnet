// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "modulated_deform_conv_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void modulated_deform_conv_forward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& bias = buildATensor(ctx, ins[2]);
  const auto& ones = buildATensor(ctx, ins[3]);
  const auto& offset = buildATensor(ctx, ins[4]);
  const auto& mask = buildATensor(ctx, ins[5]);

  auto output = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);

  modulated_deform_conv_forward(input, weight, bias, ones, offset, mask, output,
                                columns, kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, dilation_h, dilation_w, group,
                                deformable_group, with_bias);
}

void modulated_deform_conv_backward_cuda_parrots(
    CudaContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& bias = buildATensor(ctx, ins[2]);
  const auto& ones = buildATensor(ctx, ins[3]);
  const auto& offset = buildATensor(ctx, ins[4]);
  const auto& mask = buildATensor(ctx, ins[5]);

  auto columns = buildATensor(ctx, outs[0]);
  auto grad_input = buildATensor(ctx, outs[1]);
  auto grad_weight = buildATensor(ctx, outs[2]);
  auto grad_bias = buildATensor(ctx, outs[3]);
  auto grad_offset = buildATensor(ctx, outs[4]);
  auto grad_mask = buildATensor(ctx, outs[5]);
  auto grad_output = buildATensor(ctx, outs[6]);
  modulated_deform_conv_backward(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
}
#endif

void modulated_deform_conv_forward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& bias = buildATensor(ctx, ins[2]);
  const auto& ones = buildATensor(ctx, ins[3]);
  const auto& offset = buildATensor(ctx, ins[4]);
  const auto& mask = buildATensor(ctx, ins[5]);

  auto output = buildATensor(ctx, outs[0]);
  auto columns = buildATensor(ctx, outs[1]);

  modulated_deform_conv_forward(input, weight, bias, ones, offset, mask, output,
                                columns, kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, dilation_h, dilation_w, group,
                                deformable_group, with_bias);
}

void modulated_deform_conv_backward_cpu_parrots(
    HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, deformable_group, with_bias;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("stride_h", stride_h)
      .get<int>("stride_w", stride_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .get<int>("dilation_h", dilation_h)
      .get<int>("dilation_w", dilation_w)
      .get<int>("group", group)
      .get<int>("deformable_group", deformable_group)
      .get<int>("with_bias", with_bias)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& bias = buildATensor(ctx, ins[2]);
  const auto& ones = buildATensor(ctx, ins[3]);
  const auto& offset = buildATensor(ctx, ins[4]);
  const auto& mask = buildATensor(ctx, ins[5]);

  auto columns = buildATensor(ctx, outs[0]);
  auto grad_input = buildATensor(ctx, outs[1]);
  auto grad_weight = buildATensor(ctx, outs[2]);
  auto grad_bias = buildATensor(ctx, outs[3]);
  auto grad_offset = buildATensor(ctx, outs[4]);
  auto grad_mask = buildATensor(ctx, outs[5]);
  auto grad_output = buildATensor(ctx, outs[6]);
  modulated_deform_conv_backward(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
}
PARROTS_EXTENSION_REGISTER(modulated_deform_conv_forward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(2)
    .apply(modulated_deform_conv_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(modulated_deform_conv_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(modulated_deform_conv_backward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(7)
    .apply(modulated_deform_conv_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(modulated_deform_conv_backward_cuda_parrots)
#endif
    .done();

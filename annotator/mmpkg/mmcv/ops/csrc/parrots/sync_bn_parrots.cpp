// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "sync_bn_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void sync_bn_forward_mean_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
  const auto& input = buildATensor(ctx, ins[0]);
  auto mean = buildATensor(ctx, outs[0]);
  sync_bn_forward_mean_cuda(input, mean);
}

void sync_bn_forward_var_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                      const OperatorBase::in_list_t& ins,
                                      OperatorBase::out_list_t& outs) {
  const auto& input = buildATensor(ctx, ins[0]);
  const auto& mean = buildATensor(ctx, ins[1]);
  auto var = buildATensor(ctx, outs[0]);
  sync_bn_forward_var_cuda(input, mean, var);
}

void sync_bn_forward_output_cuda_parrots(CudaContext& ctx,
                                         const SSElement& attr,
                                         const OperatorBase::in_list_t& ins,
                                         OperatorBase::out_list_t& outs) {
  size_t group_size;
  float eps, momentum;
  SSAttrs(attr)
      .get<float>("eps", eps)
      .get<float>("momentum", momentum)
      .get<size_t>("group_size", group_size)
      .done();

  const auto& input = buildATensor(ctx, ins[0]);
  const auto& mean = buildATensor(ctx, ins[1]);
  const auto& var = buildATensor(ctx, ins[2]);
  const auto& weight = buildATensor(ctx, ins[3]);
  const auto& bias = buildATensor(ctx, ins[4]);
  auto running_mean = buildATensor(ctx, outs[0]);
  auto running_var = buildATensor(ctx, outs[1]);
  auto norm = buildATensor(ctx, outs[2]);
  auto std = buildATensor(ctx, outs[3]);
  auto output = buildATensor(ctx, outs[4]);
  sync_bn_forward_output_cuda(input, mean, var, running_mean, running_var,
                              weight, bias, norm, std, output, eps, momentum,
                              group_size);
}

void sync_bn_backward_param_cuda_parrots(CudaContext& ctx,
                                         const SSElement& attr,
                                         const OperatorBase::in_list_t& ins,
                                         OperatorBase::out_list_t& outs) {
  const auto& grad_output = buildATensor(ctx, ins[0]);
  const auto& norm = buildATensor(ctx, ins[1]);
  auto grad_weight = buildATensor(ctx, outs[0]);
  auto grad_bias = buildATensor(ctx, outs[1]);
  sync_bn_backward_param_cuda(grad_output, norm, grad_weight, grad_bias);
}

void sync_bn_backward_data_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  const auto& grad_output = buildATensor(ctx, ins[0]);
  const auto& weight = buildATensor(ctx, ins[1]);
  const auto& grad_weight = buildATensor(ctx, ins[2]);
  const auto& grad_bias = buildATensor(ctx, ins[3]);
  const auto& norm = buildATensor(ctx, ins[4]);
  const auto& std = buildATensor(ctx, ins[5]);
  auto grad_input = buildATensor(ctx, outs[0]);
  sync_bn_backward_data_cuda(grad_output, weight, grad_weight, grad_bias, norm,
                             std, grad_input);
}

PARROTS_EXTENSION_REGISTER(sync_bn_forward_mean)
    .input(1)
    .output(1)
    .apply(sync_bn_forward_mean_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(sync_bn_forward_var)
    .input(2)
    .output(1)
    .apply(sync_bn_forward_var_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(sync_bn_forward_output)
    .attr("eps")
    .attr("momentum")
    .attr("group_size")
    .input(5)
    .output(5)
    .apply(sync_bn_forward_output_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(sync_bn_backward_param)
    .input(2)
    .output(2)
    .apply(sync_bn_backward_param_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(sync_bn_backward_data)
    .input(6)
    .output(1)
    .apply(sync_bn_backward_data_cuda_parrots)
    .done();
#endif

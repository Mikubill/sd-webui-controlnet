// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "psamask_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void psamask_forward_cuda_parrots(CudaContext &ctx, const SSElement &attr,
                                  const OperatorBase::in_list_t &ins,
                                  OperatorBase::out_list_t &outs) {
  int psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
      half_w_mask;
  SSAttrs(attr)
      .get<int>("psa_type", psa_type)
      .get<int>("num_", num_)
      .get<int>("h_feature", h_feature)
      .get<int>("w_feature", w_feature)
      .get<int>("h_mask", h_mask)
      .get<int>("w_mask", w_mask)
      .get<int>("half_h_mask", half_h_mask)
      .get<int>("half_w_mask", half_w_mask)
      .done();
  const auto &input = buildATensor(ctx, ins[0]);
  auto output = buildATensor(ctx, outs[0]);
  psamask_forward_cuda(psa_type, input, output, num_, h_feature, w_feature,
                       h_mask, w_mask, half_h_mask, half_w_mask);
}

void psamask_backward_cuda_parrots(CudaContext &ctx, const SSElement &attr,
                                   const OperatorBase::in_list_t &ins,
                                   OperatorBase::out_list_t &outs) {
  int psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
      half_w_mask;
  SSAttrs(attr)
      .get<int>("psa_type", psa_type)
      .get<int>("num_", num_)
      .get<int>("h_feature", h_feature)
      .get<int>("w_feature", w_feature)
      .get<int>("h_mask", h_mask)
      .get<int>("w_mask", w_mask)
      .get<int>("half_h_mask", half_h_mask)
      .get<int>("half_w_mask", half_w_mask)
      .done();

  const auto &grad_output = buildATensor(ctx, ins[0]);
  auto grad_input = buildATensor(ctx, outs[0]);
  psamask_backward_cuda(psa_type, grad_output, grad_input, num_, h_feature,
                        w_feature, h_mask, w_mask, half_h_mask, half_w_mask);
}
#endif

void psamask_forward_cpu_parrots(HostContext &ctx, const SSElement &attr,
                                 const OperatorBase::in_list_t &ins,
                                 OperatorBase::out_list_t &outs) {
  int psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
      half_w_mask;
  SSAttrs(attr)
      .get<int>("psa_type", psa_type)
      .get<int>("num_", num_)
      .get<int>("h_feature", h_feature)
      .get<int>("w_feature", w_feature)
      .get<int>("h_mask", h_mask)
      .get<int>("w_mask", w_mask)
      .get<int>("half_h_mask", half_h_mask)
      .get<int>("half_w_mask", half_w_mask)
      .done();
  const auto &input = buildATensor(ctx, ins[0]);
  auto output = buildATensor(ctx, outs[0]);
  psamask_forward_cpu(psa_type, input, output, num_, h_feature, w_feature,
                      h_mask, w_mask, half_h_mask, half_w_mask);
}

void psamask_backward_cpu_parrots(HostContext &ctx, const SSElement &attr,
                                  const OperatorBase::in_list_t &ins,
                                  OperatorBase::out_list_t &outs) {
  int psa_type, num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
      half_w_mask;
  SSAttrs(attr)
      .get<int>("psa_type", psa_type)
      .get<int>("num_", num_)
      .get<int>("h_feature", h_feature)
      .get<int>("w_feature", w_feature)
      .get<int>("h_mask", h_mask)
      .get<int>("w_mask", w_mask)
      .get<int>("half_h_mask", half_h_mask)
      .get<int>("half_w_mask", half_w_mask)
      .done();

  const auto &grad_output = buildATensor(ctx, ins[0]);
  auto grad_input = buildATensor(ctx, outs[0]);
  psamask_backward_cpu(psa_type, grad_output, grad_input, num_, h_feature,
                       w_feature, h_mask, w_mask, half_h_mask, half_w_mask);
}

PARROTS_EXTENSION_REGISTER(psamask_forward)
    .attr("psa_type")
    .attr("num_")
    .attr("h_feature")
    .attr("w_feature")
    .attr("h_mask")
    .attr("w_mask")
    .attr("half_h_mask")
    .attr("half_w_mask")
    .input(1)
    .output(1)
    .apply(psamask_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(psamask_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(psamask_backward)
    .attr("psa_type")
    .attr("num_")
    .attr("h_feature")
    .attr("w_feature")
    .attr("h_mask")
    .attr("w_mask")
    .attr("half_h_mask")
    .attr("half_w_mask")
    .input(1)
    .output(1)
    .apply(psamask_backward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(psamask_backward_cuda_parrots)
#endif
    .done();

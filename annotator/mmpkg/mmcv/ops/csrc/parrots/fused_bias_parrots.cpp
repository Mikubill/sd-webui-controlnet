// Copyright (c) OpenMMLab. All rights reserved
#include <torch/extension.h>

#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
using namespace at;
using namespace parrots;

torch::Tensor fused_bias_leakyrelu(const torch::Tensor &input,
                                   const torch::Tensor &bias,
                                   const torch::Tensor &refer, int act,
                                   int grad, float alpha, float scale);

void fused_bias_leakyrelu_parrots(CudaContext &ctx, const SSElement &attr,
                                  const OperatorBase::in_list_t &ins,
                                  OperatorBase::out_list_t &outs) {
  int act, grad;
  float alpha, scale;
  SSAttrs(attr)
      .get<int>("act", act)
      .get<int>("grad", grad)
      .get<float>("alpha", alpha)
      .get<float>("scale", scale)
      .done();
  const auto &input = buildATensor(ctx, ins[0]);
  const auto &bias = buildATensor(ctx, ins[1]);
  const auto &refer = buildATensor(ctx, ins[2]);
  auto out = fused_bias_leakyrelu(input, bias, refer, act, grad, alpha, scale);
  updateDArray(ctx, out, outs[0]);
}

PARROTS_EXTENSION_REGISTER(fused_bias_leakyrelu)
    .attr("act")
    .attr("grad")
    .attr("alpha")
    .attr("scale")
    .input(3)
    .output(1)
    .apply(fused_bias_leakyrelu_parrots)
    .done();

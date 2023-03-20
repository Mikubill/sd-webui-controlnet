// Copyright (c) OpenMMLab. All rights reserved
#include <torch/extension.h>

#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
using namespace at;
using namespace parrots;

torch::Tensor upfirdn2d(const Tensor &input, const Tensor &kernel, int up_x,
                        int up_y, int down_x, int down_y, int pad_x0,
                        int pad_x1, int pad_y0, int pad_y1);

void upfirdn2d_parrots(CudaContext &ctx, const SSElement &attr,
                       const OperatorBase::in_list_t &ins,
                       OperatorBase::out_list_t &outs) {
  int up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1;
  const auto &input = buildATensor(ctx, ins[0]);
  const auto &kernel = buildATensor(ctx, ins[1]);
  SSAttrs(attr)
      .get("up_x", up_x)
      .get("up_y", up_y)
      .get("down_x", down_x)
      .get("down_y", down_y)
      .get("pad_x0", pad_x0)
      .get("pad_x1", pad_x1)
      .get("pad_y0", pad_y0)
      .get("pad_y1", pad_y1)
      .done();
  auto out = upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0,
                       pad_x1, pad_y0, pad_y1);
  updateDArray(ctx, out, outs[0]);
}

PARROTS_EXTENSION_REGISTER(upfirdn2d)
    .attr("up_x")
    .attr("up_y")
    .attr("down_x")
    .attr("down_y")
    .attr("pad_x0")
    .attr("pad_x1")
    .attr("pad_y0")
    .attr("pad_y1")
    .input(2)
    .output(1)
    .apply(upfirdn2d_parrots)
    .done();

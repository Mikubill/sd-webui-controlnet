// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "contour_expand_pytorch.h"

using namespace parrots;
using namespace std;

template <typename T>
void contour_expand_parrots(T& ctx, const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
  int min_kernel_area, kernel_num;
  SSAttrs(attr)
      .get<int>("min_kernel_area", min_kernel_area)
      .get<int>("kernel_num", kernel_num)
      .done();
  at::Tensor kernel_mask;
  at::Tensor internal_kernel_label;
  kernel_mask = buildATensor(ctx, ins[0]);
  internal_kernel_label = buildATensor(ctx, ins[1]);
  auto out = contour_expand(kernel_mask, internal_kernel_label, min_kernel_area,
                            kernel_num);
  int n = out.size(), m = 0;
  for (int i = 0; i < n; ++i)
    if (m < out[i].size()) m = out[i].size();
  auto options = torch::TensorOptions().dtype(at::kInt);
  auto tensor = torch::zeros({n, m}, options);
  for (int i = 0; i < n; i++)
    tensor.slice(0, i, i + 1) =
        torch::from_blob(out[i].data(), {out[i].size()}, options);
  updateDArray(ctx, tensor, outs[0]);
}

PARROTS_EXTENSION_REGISTER(contour_expand)
    .attr("min_kernel_area")
    .attr("kernel_num")
    .input(2)
    .output(1)
    .apply(contour_expand_parrots<HostContext>)
    .done();

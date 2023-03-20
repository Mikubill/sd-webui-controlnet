// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "pixel_group_pytorch.h"

using namespace parrots;
using namespace std;

template <typename T>
void pixel_group_parrots(T& ctx, const SSElement& attr,
                         const OperatorBase::in_list_t& ins,
                         OperatorBase::out_list_t& outs) {
  int kernel_region_num;
  float distance_threshold;
  SSAttrs(attr)
      .get<int>("kernel_region_num", kernel_region_num)
      .get<float>("distance_threshold", distance_threshold)
      .done();
  at::Tensor score;
  at::Tensor mask;
  at::Tensor embedding;
  at::Tensor kernel_label;
  at::Tensor kernel_contour;
  score = buildATensor(ctx, ins[0]);
  mask = buildATensor(ctx, ins[1]);
  embedding = buildATensor(ctx, ins[2]);
  kernel_label = buildATensor(ctx, ins[3]);
  kernel_contour = buildATensor(ctx, ins[4]);
  auto out = pixel_group(score, mask, embedding, kernel_label, kernel_contour,
                         kernel_region_num, distance_threshold);
  int n = out.size();
  std::vector<float> out_tensor;
  for (int i = 0; i < n; ++i) out_tensor.push_back(float(out[i].size()));
  for (int i = 0; i < n; ++i)
    out_tensor.insert(out_tensor.end(), out[i].begin(), out[i].end());
  auto options = torch::TensorOptions().dtype(at::kFloat);
  auto tensor = torch::zeros({1, out_tensor.size()}, options);
  tensor.slice(0, 0, 1) =
      torch::from_blob(out_tensor.data(), {out_tensor.size()}, options);
  updateDArray(ctx, tensor, outs[0]);
}

PARROTS_EXTENSION_REGISTER(pixel_group)
    .attr("kernel_region_num")
    .attr("distance_threshold")
    .input(5)
    .output(1)
    .apply(pixel_group_parrots<HostContext>)
#ifdef MMCV_WITH_CUDA
    .apply(pixel_group_parrots<CudaContext>)
#endif
    .done();

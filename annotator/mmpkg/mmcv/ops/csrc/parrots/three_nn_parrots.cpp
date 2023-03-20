// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "three_nn_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void three_nn_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                   const OperatorBase::in_list_t& ins,
                                   OperatorBase::out_list_t& outs) {
  int b, n, m;
  SSAttrs(attr).get<int>("b", b).get<int>("n", n).get<int>("m", m).done();

  auto unknown_tensor = buildATensor(ctx, ins[0]);
  auto known_tensor = buildATensor(ctx, ins[1]);

  auto dist2_tensor = buildATensor(ctx, outs[0]);
  auto idx_tensor = buildATensor(ctx, outs[1]);

  three_nn_forward(unknown_tensor, known_tensor, dist2_tensor, idx_tensor, b, n,
                   m);
}

PARROTS_EXTENSION_REGISTER(three_nn_forward)
    .attr("b")
    .attr("n")
    .attr("m")
    .input(2)
    .output(2)
    .apply(three_nn_forward_cuda_parrots)
    .done();
#endif

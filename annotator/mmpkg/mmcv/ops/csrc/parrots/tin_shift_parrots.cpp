// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "tin_shift_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void tin_shift_forward_cuda_parrots(CudaContext &ctx, const SSElement &attr,
                                    const OperatorBase::in_list_t &ins,
                                    OperatorBase::out_list_t &outs) {
  const auto &input = buildATensor(ctx, ins[0]);
  const auto &shift = buildATensor(ctx, ins[1]);
  auto output = buildATensor(ctx, outs[0]);
  tin_shift_forward_cuda(input, shift, output);
}

void tin_shift_backward_cuda_parrots(CudaContext &ctx, const SSElement &attr,
                                     const OperatorBase::in_list_t &ins,
                                     OperatorBase::out_list_t &outs) {
  const auto &grad_output = buildATensor(ctx, ins[0]);
  const auto &shift = buildATensor(ctx, ins[1]);
  auto grad_input = buildATensor(ctx, outs[0]);
  tin_shift_backward_cuda(grad_output, shift, grad_input);
}

PARROTS_EXTENSION_REGISTER(tin_shift_forward)
    .input(2)
    .output(1)
    .apply(tin_shift_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(tin_shift_backward)
    .input(2)
    .output(1)
    .apply(tin_shift_backward_cuda_parrots)
    .done();
#endif

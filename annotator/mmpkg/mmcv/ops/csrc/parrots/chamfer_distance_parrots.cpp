// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "chamfer_distance_pytorch.h"
using namespace parrots;

#ifdef MMCV_WITH_CUDA
void chamfer_distance_forward_cuda_parrots(CudaContext& ctx,
                                           const SSElement& attr,
                                           const OperatorBase::in_list_t& ins,
                                           OperatorBase::out_list_t& outs) {
  auto xyz1 = buildATensor(ctx, ins[0]);
  auto xyz2 = buildATensor(ctx, ins[1]);
  auto dist1 = buildATensor(ctx, outs[0]);
  auto dist2 = buildATensor(ctx, outs[1]);
  auto idx1 = buildATensor(ctx, outs[2]);
  auto idx2 = buildATensor(ctx, outs[3]);
  chamfer_distance_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);
}

void chamfer_distance_backward_cuda_parrots(CudaContext& ctx,
                                            const SSElement& attr,
                                            const OperatorBase::in_list_t& ins,
                                            OperatorBase::out_list_t& outs) {
  auto xyz1 = buildATensor(ctx, ins[0]);
  auto xyz2 = buildATensor(ctx, ins[1]);
  auto idx1 = buildATensor(ctx, ins[2]);
  auto idx2 = buildATensor(ctx, ins[3]);
  auto graddist1 = buildATensor(ctx, ins[4]);
  auto graddist2 = buildATensor(ctx, ins[5]);
  auto gradxyz1 = buildATensor(ctx, outs[0]);
  auto gradxyz2 = buildATensor(ctx, outs[1]);
  chamfer_distance_backward(xyz1, xyz2, idx1, idx2, graddist1, graddist2,
                            gradxyz1, gradxyz2);
}

PARROTS_EXTENSION_REGISTER(chamfer_distance_forward)
    .input(2)
    .output(4)
    .apply(chamfer_distance_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(chamfer_distance_backward)
    .input(6)
    .output(2)
    .apply(chamfer_distance_backward_cuda_parrots)
    .done();

#endif

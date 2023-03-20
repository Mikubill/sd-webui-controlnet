// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "roipoint_pool3d_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void roipoint_pool3d_forward_cuda_parrots(CudaContext& ctx,
                                          const SSElement& attr,
                                          const OperatorBase::in_list_t& ins,
                                          OperatorBase::out_list_t& outs) {
  auto xyz = buildATensor(ctx, ins[0]);
  auto boxes3d = buildATensor(ctx, ins[1]);
  auto pts_feature = buildATensor(ctx, ins[2]);

  auto pooled_features = buildATensor(ctx, outs[0]);
  auto pooled_empty_flag = buildATensor(ctx, outs[1]);

  roipoint_pool3d_forward(xyz, boxes3d, pts_feature, pooled_features,
                          pooled_empty_flag);
}

PARROTS_EXTENSION_REGISTER(roipoint_pool3d_forward)
    .input(3)
    .output(2)
    .apply(roipoint_pool3d_forward_cuda_parrots)
    .done();
#endif

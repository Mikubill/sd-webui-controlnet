// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "roiaware_pool3d_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void roiaware_pool3d_forward_cuda_parrots(CudaContext& ctx,
                                          const SSElement& attr,
                                          const OperatorBase::in_list_t& ins,
                                          OperatorBase::out_list_t& outs) {
  int pool_method;
  SSAttrs(attr).get<int>("pool_method", pool_method).done();
  auto rois = buildATensor(ctx, ins[0]);
  auto pts = buildATensor(ctx, ins[1]);
  auto pts_feature = buildATensor(ctx, ins[2]);

  auto argmax = buildATensor(ctx, outs[0]);
  auto pts_idx_of_voxels = buildATensor(ctx, outs[1]);
  auto pooled_features = buildATensor(ctx, outs[2]);

  roiaware_pool3d_forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels,
                          pooled_features, pool_method);
}

void roiaware_pool3d_backward_cuda_parrots(CudaContext& ctx,
                                           const SSElement& attr,
                                           const OperatorBase::in_list_t& ins,
                                           OperatorBase::out_list_t& outs) {
  int pool_method;
  SSAttrs(attr).get<int>("pool_method", pool_method).done();
  auto pts_idx_of_voxels = buildATensor(ctx, ins[0]);
  auto argmax = buildATensor(ctx, ins[1]);
  auto grad_out = buildATensor(ctx, ins[2]);

  auto grad_in = buildATensor(ctx, outs[0]);

  roiaware_pool3d_backward(pts_idx_of_voxels, argmax, grad_out, grad_in,
                           pool_method);
}

PARROTS_EXTENSION_REGISTER(roiaware_pool3d_forward)
    .attr("pool_method")
    .input(3)
    .output(3)
    .apply(roiaware_pool3d_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(roiaware_pool3d_backward)
    .attr("pool_method")
    .input(3)
    .output(1)
    .apply(roiaware_pool3d_backward_cuda_parrots)
    .done();
#endif

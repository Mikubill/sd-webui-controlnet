// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "voxelization_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void hard_voxelize_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  int max_points, max_voxels, NDim;
  bool deterministic;
  SSAttrs(attr)
      .get<int>("max_points", max_points)
      .get<int>("max_voxels", max_voxels)
      .get<int>("NDim", NDim)
      .get<bool>("deterministic", deterministic)
      .done();
  const auto& points = buildATensor(ctx, ins[0]);
  const auto& voxel_size = buildATensor(ctx, ins[1]);
  const auto& coors_range = buildATensor(ctx, ins[2]);

  auto voxels = buildATensor(ctx, outs[0]);
  auto coors = buildATensor(ctx, outs[1]);
  auto num_points_per_voxel = buildATensor(ctx, outs[2]);
  auto voxel_num = buildATensor(ctx, outs[3]);

  hard_voxelize_forward(points, voxel_size, coors_range, voxels, coors,
                        num_points_per_voxel, voxel_num, max_points, max_voxels,
                        NDim, deterministic);
}

void dynamic_voxelize_forward_cuda_parrots(CudaContext& ctx,
                                           const SSElement& attr,
                                           const OperatorBase::in_list_t& ins,
                                           OperatorBase::out_list_t& outs) {
  int NDim;
  SSAttrs(attr).get<int>("NDim", NDim).done();
  const auto& points = buildATensor(ctx, ins[0]);
  const auto& voxel_size = buildATensor(ctx, ins[1]);
  const auto& coors_range = buildATensor(ctx, ins[2]);

  auto coors = buildATensor(ctx, outs[0]);

  dynamic_voxelize_forward(points, voxel_size, coors_range, coors, NDim);
}
#endif

void hard_voxelize_forward_cpu_parrots(HostContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
  int max_points, max_voxels, NDim;
  bool deterministic;
  SSAttrs(attr)
      .get<int>("max_points", max_points)
      .get<int>("max_voxels", max_voxels)
      .get<int>("NDim", NDim)
      .get<bool>("deterministic", deterministic)
      .done();
  const auto& points = buildATensor(ctx, ins[0]);
  const auto& voxel_size = buildATensor(ctx, ins[1]);
  const auto& coors_range = buildATensor(ctx, ins[2]);

  auto voxels = buildATensor(ctx, outs[0]);
  auto coors = buildATensor(ctx, outs[1]);
  auto num_points_per_voxel = buildATensor(ctx, outs[2]);
  auto voxel_num = buildATensor(ctx, outs[3]);

  hard_voxelize_forward(points, voxel_size, coors_range, voxels, coors,
                        num_points_per_voxel, voxel_num, max_points, max_voxels,
                        NDim, deterministic);
}

void dynamic_voxelize_forward_cpu_parrots(HostContext& ctx,
                                          const SSElement& attr,
                                          const OperatorBase::in_list_t& ins,
                                          OperatorBase::out_list_t& outs) {
  int NDim;
  SSAttrs(attr).get<int>("NDim", NDim).done();
  const auto& points = buildATensor(ctx, ins[0]);
  const auto& voxel_size = buildATensor(ctx, ins[1]);
  const auto& coors_range = buildATensor(ctx, ins[2]);

  auto coors = buildATensor(ctx, outs[0]);

  dynamic_voxelize_forward(points, voxel_size, coors_range, coors, NDim);
}

PARROTS_EXTENSION_REGISTER(hard_voxelize_forward)
    .attr("max_points")
    .attr("max_voxels")
    .attr("NDim")
    .attr("deterministic")
    .input(3)
    .output(4)
    .apply(hard_voxelize_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(hard_voxelize_forward_cuda_parrots)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(dynamic_voxelize_forward)
    .attr("NDim")
    .input(3)
    .output(1)
    .apply(dynamic_voxelize_forward_cpu_parrots)
#ifdef MMCV_WITH_CUDA
    .apply(dynamic_voxelize_forward_cuda_parrots)
#endif
    .done();

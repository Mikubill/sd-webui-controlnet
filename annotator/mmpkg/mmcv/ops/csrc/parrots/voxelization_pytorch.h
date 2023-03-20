// Copyright (c) OpenMMLab. All rights reserved
#ifndef VOXELIZATION_PYTORCH_H
#define VOXELIZATION_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void hard_voxelize_forward(const at::Tensor &points,
                           const at::Tensor &voxel_size,
                           const at::Tensor &coors_range, at::Tensor &voxels,
                           at::Tensor &coors, at::Tensor &num_points_per_voxel,
                           at::Tensor &voxel_num, const int max_points,
                           const int max_voxels, const int NDim = 3,
                           const bool deterministic = true);

void dynamic_voxelize_forward(const at::Tensor &points,
                              const at::Tensor &voxel_size,
                              const at::Tensor &coors_range, at::Tensor &coors,
                              const int NDim = 3);

#endif  // VOXELIZATION_PYTORCH_H

// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <stdio.h>

#include "pytorch_cuda_helper.hpp"
#include "roiaware_pool3d_cuda_kernel.cuh"

void RoiawarePool3dForwardCUDAKernelLauncher(
    int boxes_num, int pts_num, int channels, int max_pts_each_voxel, int out_x,
    int out_y, int out_z, const Tensor rois, const Tensor pts,
    const Tensor pts_feature, Tensor argmax, Tensor pts_idx_of_voxels,
    Tensor pooled_features, int pool_method) {
  // params rois: (N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate params pts: (npoints, 3) [x, y, z] in LiDAR coordinate params
  // pts_feature: (npoints, C) params argmax: (N, out_x, out_y, out_z, C) params
  // pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel) params
  // pooled_features: (N, out_x, out_y, out_z, C) params pool_method: 0:
  // max_pool 1: avg_pool

  at::cuda::CUDAGuard device_guard(pts_feature.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  Tensor pts_mask =
      -at::ones({boxes_num, pts_num}, pts_feature.options().dtype(at::kInt));

  dim3 blocks_mask(GET_BLOCKS(pts_num, THREADS_PER_BLOCK), boxes_num);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      rois.scalar_type(), "generate_pts_mask_for_box3d", [&] {
        generate_pts_mask_for_box3d<scalar_t>
            <<<blocks_mask, threads, 0, stream>>>(
                boxes_num, pts_num, out_x, out_y, out_z,
                rois.data_ptr<scalar_t>(), pts.data_ptr<scalar_t>(),
                pts_mask.data_ptr<int>());
      });

  AT_CUDA_CHECK(cudaGetLastError());

  // TODO: Merge the collect and pool functions, SS

  dim3 blocks_collect(GET_BLOCKS(boxes_num, THREADS_PER_BLOCK));

  AT_DISPATCH_INTEGRAL_TYPES(
      pts_idx_of_voxels.scalar_type(), "collect_inside_pts_for_box3d", [&] {
        collect_inside_pts_for_box3d<scalar_t>
            <<<blocks_collect, threads, 0, stream>>>(
                boxes_num, pts_num, max_pts_each_voxel, out_x, out_y, out_z,
                pts_mask.data_ptr<int>(),
                pts_idx_of_voxels.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());

  dim3 blocks_pool(GET_BLOCKS(out_x * out_y * out_z, THREADS_PER_BLOCK),
                   channels, boxes_num);
  if (pool_method == 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        pts_feature.scalar_type(), "roiaware_maxpool3d", [&] {
          roiaware_maxpool3d<scalar_t><<<blocks_pool, threads, 0, stream>>>(
              boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y,
              out_z, pts_feature.data_ptr<scalar_t>(),
              pts_idx_of_voxels.data_ptr<int>(),
              pooled_features.data_ptr<scalar_t>(), argmax.data_ptr<int>());
        });
  } else if (pool_method == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        pts_feature.scalar_type(), "roiaware_avgpool3d", [&] {
          roiaware_avgpool3d<scalar_t><<<blocks_pool, threads, 0, stream>>>(
              boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y,
              out_z, pts_feature.data_ptr<scalar_t>(),
              pts_idx_of_voxels.data_ptr<int>(),
              pooled_features.data_ptr<scalar_t>());
        });
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

void RoiawarePool3dBackwardCUDAKernelLauncher(
    int boxes_num, int out_x, int out_y, int out_z, int channels,
    int max_pts_each_voxel, const Tensor pts_idx_of_voxels, const Tensor argmax,
    const Tensor grad_out, Tensor grad_in, int pool_method) {
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value
  // params pool_method: 0: max_pool, 1: avg_pool

  at::cuda::CUDAGuard device_guard(grad_out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(GET_BLOCKS(out_x * out_y * out_z, THREADS_PER_BLOCK), channels,
              boxes_num);
  dim3 threads(THREADS_PER_BLOCK);

  if (pool_method == 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_in.scalar_type(), "roiaware_maxpool3d_backward", [&] {
          roiaware_maxpool3d_backward<scalar_t><<<blocks, threads, 0, stream>>>(
              boxes_num, channels, out_x, out_y, out_z, argmax.data_ptr<int>(),
              grad_out.data_ptr<scalar_t>(), grad_in.data_ptr<scalar_t>());
        });
  } else if (pool_method == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_in.scalar_type(), "roiaware_avgpool3d_backward", [&] {
          roiaware_avgpool3d_backward<scalar_t><<<blocks, threads, 0, stream>>>(
              boxes_num, channels, out_x, out_y, out_z, max_pts_each_voxel,
              pts_idx_of_voxels.data_ptr<int>(), grad_out.data_ptr<scalar_t>(),
              grad_in.data_ptr<scalar_t>());
        });
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

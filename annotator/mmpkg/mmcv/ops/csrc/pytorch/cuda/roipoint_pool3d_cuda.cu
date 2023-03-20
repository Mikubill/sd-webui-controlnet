/*
Modified from
https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d_kernel.cu
Point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <math.h>
#include <stdio.h>

#include "pytorch_cuda_helper.hpp"
#include "roipoint_pool3d_cuda_kernel.cuh"

void RoIPointPool3dForwardCUDAKernelLauncher(
    int batch_size, int pts_num, int boxes_num, int feature_in_len,
    int sampled_pts_num, const Tensor xyz, const Tensor boxes3d,
    const Tensor pts_feature, Tensor pooled_features,
    Tensor pooled_empty_flag) {
  Tensor pts_assign = at::empty({batch_size, pts_num, boxes_num},
                                boxes3d.options().dtype(at::kInt));

  at::cuda::CUDAGuard device_guard(xyz.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      xyz.scalar_type(), "assign_pts_to_box3d", [&] {
        assign_pts_to_box3d<scalar_t><<<blocks, threads, 0, stream>>>(
            batch_size, pts_num, boxes_num, xyz.data_ptr<scalar_t>(),
            boxes3d.data_ptr<scalar_t>(), pts_assign.data_ptr<int>());
      });

  Tensor pts_idx = at::empty({batch_size, boxes_num, sampled_pts_num},
                             boxes3d.options().dtype(at::kInt));

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks2(GET_BLOCKS(boxes_num, THREADS_PER_BLOCK), batch_size);

  get_pooled_idx<<<blocks2, threads, 0, stream>>>(
      batch_size, pts_num, boxes_num, sampled_pts_num,
      pts_assign.data_ptr<int>(), pts_idx.data_ptr<int>(),
      pooled_empty_flag.data_ptr<int>());

  dim3 blocks_pool(GET_BLOCKS(sampled_pts_num, THREADS_PER_BLOCK), boxes_num,
                   batch_size);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      xyz.scalar_type(), "roipoint_pool3d_forward", [&] {
        roipoint_pool3d_forward<scalar_t><<<blocks_pool, threads, 0, stream>>>(
            batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
            xyz.data_ptr<scalar_t>(), pts_idx.data_ptr<int>(),
            pts_feature.data_ptr<scalar_t>(),
            pooled_features.data_ptr<scalar_t>(),
            pooled_empty_flag.data_ptr<int>());
      });
}

// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu
// Written by Shaoshuai Shi
// All Rights Reserved 2019.

#include <stdio.h>

#include "points_in_boxes_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void PointsInBoxesPartForwardCUDAKernelLauncher(int batch_size, int boxes_num,
                                                int pts_num, const Tensor boxes,
                                                const Tensor pts,
                                                Tensor box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is
  // the bottom center, each box DO NOT overlaps params pts: (B, npoints, 3) [x,
  // y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default
  // -1

  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(GET_BLOCKS(pts_num, THREADS_PER_BLOCK), batch_size);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      boxes.scalar_type(), "points_in_boxes_part_forward_cuda_kernel", [&] {
        points_in_boxes_part_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                batch_size, boxes_num, pts_num, boxes.data_ptr<scalar_t>(),
                pts.data_ptr<scalar_t>(), box_idx_of_points.data_ptr<int>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void PointsInBoxesAllForwardCUDAKernelLauncher(int batch_size, int boxes_num,
                                               int pts_num, const Tensor boxes,
                                               const Tensor pts,
                                               Tensor box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center, each box params pts: (B, npoints, 3)
  // [x, y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints),
  // default -1

  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(GET_BLOCKS(pts_num, THREADS_PER_BLOCK), batch_size);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      boxes.scalar_type(), "points_in_boxes_all_forward_cuda_kernel", [&] {
        points_in_boxes_all_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                batch_size, boxes_num, pts_num, boxes.data_ptr<scalar_t>(),
                pts.data_ptr<scalar_t>(), box_idx_of_points.data_ptr<int>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

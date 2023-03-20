// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/ming71/CUDA/blob/master/point_justify/points_justify_kernel.cu

#include <stdio.h>

#include "points_in_polygons_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void PointsInPolygonsForwardCUDAKernelLauncher(const at::Tensor points,
                                               const at::Tensor polygons,
                                               const int rows, const int cols,
                                               at::Tensor output) {
  const int output_size = rows * cols;
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "points_in_polygons_forward_cuda_kernel", ([&] {
        const scalar_t *vertex1 = points.data_ptr<scalar_t>();
        const scalar_t *vertex2 = polygons.data_ptr<scalar_t>();
        scalar_t *inside_flag = output.data_ptr<scalar_t>();

        points_in_polygons_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, vertex1, vertex2, rows, cols, inside_flag);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

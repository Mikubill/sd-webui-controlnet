// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/ops/minareabbox/src/minareabbox_kernel.cu
#include "min_area_polygons_cuda.cuh"
#include "pytorch_cuda_helper.hpp"

void MinAreaPolygonsCUDAKernelLauncher(const Tensor pointsets,
                                       Tensor polygons) {
  int num_pointsets = pointsets.size(0);
  const int output_size = polygons.numel();
  at::cuda::CUDAGuard device_guard(pointsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      pointsets.scalar_type(), "min_area_polygons_cuda_kernel", ([&] {
        min_area_polygons_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                num_pointsets, pointsets.data_ptr<scalar_t>(),
                polygons.data_ptr<scalar_t>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

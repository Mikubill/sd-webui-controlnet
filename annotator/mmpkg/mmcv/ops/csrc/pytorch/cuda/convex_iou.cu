// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/ops/iou/src/convex_iou_kernel.cu
#include "convex_iou_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void ConvexIoUCUDAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                 Tensor ious) {
  int output_size = ious.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  at::cuda::CUDAGuard device_guard(pointsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      pointsets.scalar_type(), "convex_iou_cuda_kernel", ([&] {
        convex_iou_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK / 2, 0, stream>>>(
                num_pointsets, num_polygons, pointsets.data_ptr<scalar_t>(),
                polygons.data_ptr<scalar_t>(), ious.data_ptr<scalar_t>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void ConvexGIoUCUDAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                  Tensor output) {
  int output_size = output.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  at::cuda::CUDAGuard device_guard(pointsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      pointsets.scalar_type(), "convex_giou_cuda_kernel", ([&] {
        convex_giou_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK / 2, 0, stream>>>(
                num_pointsets, num_polygons, pointsets.data_ptr<scalar_t>(),
                polygons.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

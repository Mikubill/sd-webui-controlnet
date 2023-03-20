// Copyright (c) OpenMMLab. All rights reserved
#include "bbox_overlaps_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

// Disable fp16 on ROCm device
#ifndef MMCV_WITH_HIP
#if __CUDA_ARCH__ >= 530
template <>
__global__ void bbox_overlaps_cuda_kernel<at::Half>(
    const at::Half* bbox1, const at::Half* bbox2, at::Half* ious,
    const int num_bbox1, const int num_bbox2, const int mode,
    const bool aligned, const int offset) {
  bbox_overlaps_cuda_kernel_half(reinterpret_cast<const __half*>(bbox1),
                                 reinterpret_cast<const __half*>(bbox2),
                                 reinterpret_cast<__half*>(ious), num_bbox1,
                                 num_bbox2, mode, aligned, offset);
}

#endif  // __CUDA_ARCH__ >= 530
#endif  // MMCV_WITH_HIP

void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset) {
  int output_size = ious.numel();
  int num_bbox1 = bboxes1.size(0);
  int num_bbox2 = bboxes2.size(0);

  at::cuda::CUDAGuard device_guard(bboxes1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bboxes1.scalar_type(), "bbox_overlaps_cuda_kernel", ([&] {
        bbox_overlaps_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                bboxes1.data_ptr<scalar_t>(), bboxes2.data_ptr<scalar_t>(),
                ious.data_ptr<scalar_t>(), num_bbox1, num_bbox2, mode, aligned,
                offset);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

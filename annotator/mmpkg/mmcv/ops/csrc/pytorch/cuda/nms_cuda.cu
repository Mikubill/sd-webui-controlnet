// Copyright (c) OpenMMLab. All rights reserved
#include "nms_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset) {
  at::cuda::CUDAGuard device_guard(boxes.device());

  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }
  auto order_t = std::get<1>(scores.sort(0, /*descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);
  const int col_blocks = (boxes_num + threadsPerBlock - 1) / threadsPerBlock;
  const int col_blocks_alloc = GET_BLOCKS(boxes_num, threadsPerBlock);
  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
  dim3 blocks(col_blocks_alloc, col_blocks_alloc);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  nms_cuda<<<blocks, threads, 0, stream>>>(
      boxes_num, iou_threshold, offset, boxes_sorted.data_ptr<float>(),
      (unsigned long long*)mask.data_ptr<int64_t>());

  // Filter the boxes which should be kept.
  at::Tensor keep_t = at::zeros(
      {boxes_num}, boxes.options().dtype(at::kBool).device(at::kCUDA));
  gather_keep_from_mask<<<1, min(col_blocks, THREADS_PER_BLOCK),
                          col_blocks * sizeof(unsigned long long), stream>>>(
      keep_t.data_ptr<bool>(), (unsigned long long*)mask.data_ptr<int64_t>(),
      boxes_num);
  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.masked_select(keep_t);
}

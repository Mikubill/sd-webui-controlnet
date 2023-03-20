// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <stdio.h>

#include "iou3d_cuda_kernel.cuh"
#include "nms_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(const int num_a,
                                                   const Tensor boxes_a,
                                                   const int num_b,
                                                   const Tensor boxes_b,
                                                   Tensor ans_overlap) {
  at::cuda::CUDAGuard device_guard(boxes_a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(num_b, THREADS_PER_BLOCK_IOU3D),
              GET_BLOCKS(num_a, THREADS_PER_BLOCK_IOU3D));
  dim3 threads(THREADS_PER_BLOCK_IOU3D, THREADS_PER_BLOCK_IOU3D);

  iou3d_boxes_overlap_bev_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      num_a, boxes_a.data_ptr<float>(), num_b, boxes_b.data_ptr<float>(),
      ans_overlap.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DNMS3DForwardCUDAKernelLauncher(const Tensor boxes, Tensor& keep,
                                         Tensor& keep_num,
                                         float nms_overlap_thresh) {
  using namespace at::indexing;
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int boxes_num = boxes.size(0);

  const int col_blocks =
      (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));

  dim3 blocks(GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS),
              GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  iou3d_nms3d_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(),
      (unsigned long long*)mask.data_ptr<int64_t>());

  at::Tensor keep_t = at::zeros(
      {boxes_num}, boxes.options().dtype(at::kBool).device(at::kCUDA));
  gather_keep_from_mask<<<1, min(col_blocks, THREADS_PER_BLOCK),
                          col_blocks * sizeof(unsigned long long), stream>>>(
      keep_t.data_ptr<bool>(), (unsigned long long*)mask.data_ptr<int64_t>(),
      boxes_num);

  auto keep_data = keep_t.nonzero().index({Slice(), 0});
  keep_num.fill_(at::Scalar(keep_data.size(0)));
  keep.index_put_({Slice(0, keep_data.size(0))}, keep_data);
  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DNMS3DNormalForwardCUDAKernelLauncher(const Tensor boxes, Tensor& keep,
                                               Tensor& keep_num,
                                               float nms_overlap_thresh) {
  using namespace at::indexing;
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int boxes_num = boxes.size(0);

  const int col_blocks =
      (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));

  dim3 blocks(GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS),
              GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  iou3d_nms3d_normal_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(),
      (unsigned long long*)mask.data_ptr<int64_t>());

  at::Tensor keep_t = at::zeros(
      {boxes_num}, boxes.options().dtype(at::kBool).device(at::kCUDA));
  gather_keep_from_mask<<<1, min(col_blocks, THREADS_PER_BLOCK),
                          col_blocks * sizeof(unsigned long long), stream>>>(
      keep_t.data_ptr<bool>(), (unsigned long long*)mask.data_ptr<int64_t>(),
      boxes_num);

  auto keep_data = keep_t.nonzero().index({Slice(), 0});
  keep_num.fill_(at::Scalar(keep_data.size(0)));
  keep.index_put_({Slice(0, keep_data.size(0))}, keep_data);
  AT_CUDA_CHECK(cudaGetLastError());
}

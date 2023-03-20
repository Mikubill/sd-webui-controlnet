// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu
#include "nms_rotated_cuda.cuh"
#include "pytorch_cuda_helper.hpp"

Tensor nms_rotated_cuda(const Tensor dets, const Tensor scores,
                        const Tensor order_t, const Tensor dets_sorted,
                        float iou_threshold, const int multi_label) {
  // using scalar_t = float;
  AT_ASSERTM(dets.is_cuda(), "dets must be a CUDA tensor");
  AT_ASSERTM(scores.is_cuda(), "scores must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(dets.device());

  int dets_num = dets.size(0);

  const int col_blocks = at::cuda::ATenCeilDiv(dets_num, threadsPerBlock);

  Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.scalar_type(), "nms_rotated_kernel_cuda", [&] {
        nms_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num, iou_threshold, dets_sorted.data_ptr<scalar_t>(),
            (unsigned long long*)mask.data_ptr<int64_t>(), multi_label);
      });

  Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host =
      (unsigned long long*)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
           .to(order_t.device(), keep.scalar_type())});
}

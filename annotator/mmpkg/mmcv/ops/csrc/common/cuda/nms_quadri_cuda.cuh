// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#ifndef NMS_QUADRI_CUDA_CUH
#define NMS_QUADRI_CUDA_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif
#include "box_iou_rotated_utils.hpp"

__host__ __device__ inline int divideUP(const int x, const int y) {
  return (((x) + (y)-1) / (y));
}

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

template <typename T>
__global__ void nms_quadri_cuda_kernel(const int n_boxes,
                                       const float iou_threshold,
                                       const T* dev_boxes,
                                       unsigned long long* dev_mask,
                                       const int multi_label) {
  if (multi_label == 1) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    // Compared to nms_cuda_kernel, where each box is represented with 4 values
    // (x1, y1, x2, y2), each rotated box is represented with 8 values
    // (x1, y1, ..., x4, y4) here.
    __shared__ T block_boxes[threadsPerBlock * 8];
    if (threadIdx.x < col_size) {
      block_boxes[threadIdx.x * 8 + 0] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 0];
      block_boxes[threadIdx.x * 8 + 1] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 1];
      block_boxes[threadIdx.x * 8 + 2] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 2];
      block_boxes[threadIdx.x * 8 + 3] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 3];
      block_boxes[threadIdx.x * 8 + 4] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 4];
      block_boxes[threadIdx.x * 8 + 5] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 5];
      block_boxes[threadIdx.x * 8 + 6] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 6];
      block_boxes[threadIdx.x * 8 + 7] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 9 + 7];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
      const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
      const T* cur_box = dev_boxes + cur_box_idx * 9;
      int i = 0;
      unsigned long long t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = threadIdx.x + 1;
      }
      for (i = start; i < col_size; i++) {
        // Instead of devIoU used by original horizontal nms, here
        // we use the single_box_iou_quadri function from
        // box_iou_rotated_utils.h
        if (single_box_iou_quadri<T>(cur_box, block_boxes + i * 8, 0) >
            iou_threshold) {
          t |= 1ULL << i;
        }
      }
      const int col_blocks = divideUP(n_boxes, threadsPerBlock);
      dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
  } else {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    // Compared to nms_cuda_kernel, where each box is represented with 4 values
    // (x1, y1, x2, y2), each rotated box is represented with 8 values
    // (x1, y1, , ..., x4, y4) here.
    __shared__ T block_boxes[threadsPerBlock * 8];
    if (threadIdx.x < col_size) {
      block_boxes[threadIdx.x * 8 + 0] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 0];
      block_boxes[threadIdx.x * 8 + 1] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 1];
      block_boxes[threadIdx.x * 8 + 2] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 2];
      block_boxes[threadIdx.x * 8 + 3] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 3];
      block_boxes[threadIdx.x * 8 + 4] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 4];
      block_boxes[threadIdx.x * 8 + 5] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 5];
      block_boxes[threadIdx.x * 8 + 6] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 6];
      block_boxes[threadIdx.x * 8 + 7] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + 7];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
      const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
      const T* cur_box = dev_boxes + cur_box_idx * 8;
      int i = 0;
      unsigned long long t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = threadIdx.x + 1;
      }
      for (i = start; i < col_size; i++) {
        // Instead of devIoU used by original horizontal nms, here
        // we use the single_box_iou_quadri function from
        // box_iou_rotated_utils.h
        if (single_box_iou_quadri<T>(cur_box, block_boxes + i * 8, 0) >
            iou_threshold) {
          t |= 1ULL << i;
        }
      }
      const int col_blocks = divideUP(n_boxes, threadsPerBlock);
      dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
  }
}

#endif

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu
#ifndef NMS_ROTATED_CUDA_CUH
#define NMS_ROTATED_CUDA_CUH

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
__global__ void nms_rotated_cuda_kernel(const int n_boxes,
                                        const float iou_threshold,
                                        const T* dev_boxes,
                                        unsigned long long* dev_mask,
                                        const int multi_label) {
  // nms_rotated_cuda_kernel is modified from torchvision's nms_cuda_kernel

  if (multi_label == 1) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    // Compared to nms_cuda_kernel, where each box is represented with 4 values
    // (x1, y1, x2, y2), each rotated box is represented with 5 values
    // (x_center, y_center, width, height, angle_degrees) here.
    __shared__ T block_boxes[threadsPerBlock * 5];
    if (threadIdx.x < col_size) {
      block_boxes[threadIdx.x * 5 + 0] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 0];
      block_boxes[threadIdx.x * 5 + 1] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 1];
      block_boxes[threadIdx.x * 5 + 2] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 2];
      block_boxes[threadIdx.x * 5 + 3] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 3];
      block_boxes[threadIdx.x * 5 + 4] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 4];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
      const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
      const T* cur_box = dev_boxes + cur_box_idx * 6;
      int i = 0;
      unsigned long long t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = threadIdx.x + 1;
      }
      for (i = start; i < col_size; i++) {
        // Instead of devIoU used by original horizontal nms, here
        // we use the single_box_iou_rotated function from
        // box_iou_rotated_utils.h
        if (single_box_iou_rotated<T>(cur_box, block_boxes + i * 5, 0) >
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
    // (x1, y1, x2, y2), each rotated box is represented with 5 values
    // (x_center, y_center, width, height, angle_degrees) here.
    __shared__ T block_boxes[threadsPerBlock * 5];
    if (threadIdx.x < col_size) {
      block_boxes[threadIdx.x * 5 + 0] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
      block_boxes[threadIdx.x * 5 + 1] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
      block_boxes[threadIdx.x * 5 + 2] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
      block_boxes[threadIdx.x * 5 + 3] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
      block_boxes[threadIdx.x * 5 + 4] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
      const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
      const T* cur_box = dev_boxes + cur_box_idx * 5;
      int i = 0;
      unsigned long long t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = threadIdx.x + 1;
      }
      for (i = start; i < col_size; i++) {
        // Instead of devIoU used by original horizontal nms, here
        // we use the single_box_iou_rotated function from
        // box_iou_rotated_utils.h
        if (single_box_iou_rotated<T>(cur_box, block_boxes + i * 5, 0) >
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

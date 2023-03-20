// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#ifndef BOX_IOU_QUADRI_CUDA_CUH
#define BOX_IOU_QUADRI_CUDA_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif
#include "box_iou_rotated_utils.hpp"

// 2D block with 32 * 16 = 512 threads per block
const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 16;

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

template <typename T>
__global__ void box_iou_quadri_cuda_kernel(
    const int n_boxes1, const int n_boxes2, const T* dev_boxes1,
    const T* dev_boxes2, T* dev_ious, const int mode_flag, const bool aligned) {
  if (aligned) {
    CUDA_1D_KERNEL_LOOP(index, n_boxes1) {
      int b1 = index;
      int b2 = index;

      int base1 = b1 * 8;

      float block_boxes1[8];
      float block_boxes2[8];

      block_boxes1[0] = dev_boxes1[base1 + 0];
      block_boxes1[1] = dev_boxes1[base1 + 1];
      block_boxes1[2] = dev_boxes1[base1 + 2];
      block_boxes1[3] = dev_boxes1[base1 + 3];
      block_boxes1[4] = dev_boxes1[base1 + 4];
      block_boxes1[5] = dev_boxes1[base1 + 5];
      block_boxes1[6] = dev_boxes1[base1 + 6];
      block_boxes1[7] = dev_boxes1[base1 + 7];

      int base2 = b2 * 8;

      block_boxes2[0] = dev_boxes2[base2 + 0];
      block_boxes2[1] = dev_boxes2[base2 + 1];
      block_boxes2[2] = dev_boxes2[base2 + 2];
      block_boxes2[3] = dev_boxes2[base2 + 3];
      block_boxes2[4] = dev_boxes2[base2 + 4];
      block_boxes2[5] = dev_boxes2[base2 + 5];
      block_boxes2[6] = dev_boxes2[base2 + 6];
      block_boxes2[7] = dev_boxes2[base2 + 7];

      dev_ious[index] =
          single_box_iou_quadri<T>(block_boxes1, block_boxes2, mode_flag);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, n_boxes1 * n_boxes2) {
      int b1 = index / n_boxes2;
      int b2 = index % n_boxes2;

      int base1 = b1 * 8;

      float block_boxes1[8];
      float block_boxes2[8];

      block_boxes1[0] = dev_boxes1[base1 + 0];
      block_boxes1[1] = dev_boxes1[base1 + 1];
      block_boxes1[2] = dev_boxes1[base1 + 2];
      block_boxes1[3] = dev_boxes1[base1 + 3];
      block_boxes1[4] = dev_boxes1[base1 + 4];
      block_boxes1[5] = dev_boxes1[base1 + 5];
      block_boxes1[6] = dev_boxes1[base1 + 6];
      block_boxes1[7] = dev_boxes1[base1 + 7];

      int base2 = b2 * 8;

      block_boxes2[0] = dev_boxes2[base2 + 0];
      block_boxes2[1] = dev_boxes2[base2 + 1];
      block_boxes2[2] = dev_boxes2[base2 + 2];
      block_boxes2[3] = dev_boxes2[base2 + 3];
      block_boxes2[4] = dev_boxes2[base2 + 4];
      block_boxes2[5] = dev_boxes2[base2 + 5];
      block_boxes2[6] = dev_boxes2[base2 + 6];
      block_boxes2[7] = dev_boxes2[base2 + 7];

      dev_ious[index] =
          single_box_iou_quadri<T>(block_boxes1, block_boxes2, mode_flag);
    }
  }
}

#endif

// Copyright (c) OpenMMLab. All rights reserved
#ifndef CARAFE_NAIVE_CUDA_KERNEL_CUH
#define CARAFE_NAIVE_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

__device__ inline int Loc2Index(const int n, const int c, const int h,
                                const int w, const int channel_num,
                                const int height, const int width) {
  int index = w + (h + (c + n * channel_num) * height) * width;
  return index;
}

template <typename scalar_t>
__global__ void carafe_naive_forward_cuda_kernel(
    const int nthreads, const scalar_t *bottom_data,
    const scalar_t *bottom_masks, scalar_t *top_data, const int kernel_size,
    const int group_size, const int scale_factor, const int channels,
    const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the bottom_data
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int mask_channels = kernel_size * kernel_size * group_size;
    int mask_group = c / (channels / group_size);

    int down_pw = pw / scale_factor;
    int down_ph = ph / scale_factor;
    int down_width = width / scale_factor;
    int down_height = height / scale_factor;
    int start_w = down_pw - (kernel_size - 1) / 2;
    int end_w = down_pw + (kernel_size - 1) / 2 + 1;
    int start_h = down_ph - (kernel_size - 1) / 2;
    int end_h = down_ph + (kernel_size - 1) / 2 + 1;

    scalar_t output_val = 0;
    for (int iy = start_h; iy < end_h; iy++) {
      for (int ix = start_w; ix < end_w; ix++) {
        if (iy < 0 || iy > down_height - 1 || ix < 0 || ix > down_width - 1) {
          continue;
        }
        int mask_iy = iy - down_ph + (kernel_size - 1) / 2;
        int mask_ix = ix - down_pw + (kernel_size - 1) / 2;
        int mask_c =
            (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
        int feat_index =
            Loc2Index(n, c, iy, ix, channels, down_height, down_width);
        int mask_index =
            Loc2Index(n, mask_c, ph, pw, mask_channels, height, width);
        output_val += bottom_data[feat_index] * bottom_masks[mask_index];
      }
    }
    top_data[index] = output_val;
  }
}

template <typename scalar_t>
__global__ void carafe_naive_backward_cuda_kernel(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_data,
    const scalar_t *bottom_masks, scalar_t *bottom_diff, scalar_t *mask_diff,
    const int kernel_size, const int group_size, const int scale_factor,
    const int channels, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the bottom_data
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int mask_channels = kernel_size * kernel_size * group_size;
    int mask_group = c / (channels / group_size);

    int down_pw = pw / scale_factor;
    int down_ph = ph / scale_factor;
    int down_width = width / scale_factor;
    int down_height = height / scale_factor;
    int start_w = down_pw - (kernel_size - 1) / 2;
    int end_w = down_pw + (kernel_size - 1) / 2 + 1;
    int start_h = down_ph - (kernel_size - 1) / 2;
    int end_h = down_ph + (kernel_size - 1) / 2 + 1;

    for (int iy = start_h; iy < end_h; iy++) {
      for (int ix = start_w; ix < end_w; ix++) {
        if (iy < 0 || iy > down_height - 1 || ix < 0 || ix > down_width - 1) {
          continue;
        }
        int mask_iy = iy - down_ph + (kernel_size - 1) / 2;
        int mask_ix = ix - down_pw + (kernel_size - 1) / 2;
        int mask_c =
            (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
        int feat_index =
            Loc2Index(n, c, iy, ix, channels, down_height, down_width);
        int mask_index =
            Loc2Index(n, mask_c, ph, pw, mask_channels, height, width);
        atomicAdd(bottom_diff + feat_index,
                  bottom_masks[mask_index] * top_diff[index]);
        atomicAdd(mask_diff + mask_index,
                  bottom_data[feat_index] * top_diff[index]);
      }
    }
  }
}

#endif  // CARAFE_NAIVE_CUDA_KERNEL_CUH

// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection/blob/master/mmdet/ops/fr/src/feature_refine_kernel.cu
#ifndef ROTATED_FEATURE_ALIGN_CUDA_KERNEL_CUH
#define ROTATED_FEATURE_ALIGN_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename scalar_t>
__global__ void rotated_feature_align_forward_kernel(
    const int nthreads, const int points, const scalar_t* bottom_data,
    const scalar_t* best_bboxes, const scalar_t spatial_scale,
    const int channels, const int height, const int width, scalar_t* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    const scalar_t* bbox_offset =
        best_bboxes + ((n * height + h) * width + w) * 5;
    scalar_t roi_y = bbox_offset[0] * spatial_scale;
    scalar_t roi_x = bbox_offset[1] * spatial_scale;

    scalar_t px[5] = {roi_x, 0, 0, 0, 0};
    scalar_t py[5] = {roi_y, 0, 0, 0, 0};

    if (points > 1) {
      scalar_t roi_w = bbox_offset[2] * spatial_scale;
      scalar_t roi_h = bbox_offset[3] * spatial_scale;
      scalar_t roi_a = bbox_offset[4];

      scalar_t w_2 = roi_w / 2, h_2 = roi_h / 2;
      scalar_t cosa = cosf(roi_a), sina = sinf(roi_a);
      scalar_t wx = cosa * w_2, wy = sina * w_2;
      scalar_t hx = -sina * h_2, hy = cosa * h_2;

      px[1] = roi_x + wx + hx;
      py[1] = roi_y + wy + hy;
      px[2] = roi_x - wx + hx;
      py[2] = roi_y - wy + hy;
      px[3] = roi_x - wx - hx;
      py[3] = roi_y - wy - hy;
      px[4] = roi_x + wx - hx;
      py[4] = roi_y + wy - hy;
    }

    const scalar_t* offset_bottom_data =
        bottom_data + (n * channels + c) * height * width;

    scalar_t output_val = bottom_data[index];
    for (int i = 0; i < points; i++) {
      output_val += bilinear_interpolate<scalar_t>(offset_bottom_data, height,
                                                   width, py[i], px[i], i);
    }
    top_data[index] = output_val;
  }
}

template <typename scalar_t>
__global__ void rotated_feature_align_backward_kernel(
    const int nthreads, const int points, const scalar_t* top_diff,
    const scalar_t* best_bboxes, const scalar_t spatial_scale,
    const int channels, const int height, const int width,
    scalar_t* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    const scalar_t* bbox_offset =
        best_bboxes + ((n * height + h) * width + w) * 5;
    scalar_t roi_y = bbox_offset[0] * spatial_scale;
    scalar_t roi_x = bbox_offset[1] * spatial_scale;

    scalar_t px[5] = {roi_x, 0, 0, 0, 0};
    scalar_t py[5] = {roi_y, 0, 0, 0, 0};

    if (points > 1) {
      scalar_t roi_w = bbox_offset[2] * spatial_scale;
      scalar_t roi_h = bbox_offset[3] * spatial_scale;
      scalar_t roi_a = bbox_offset[4];

      scalar_t w_2 = roi_w / 2, h_2 = roi_h / 2;
      scalar_t cosa = cosf(roi_a), sina = sinf(roi_a);
      scalar_t wx = cosa * w_2, wy = sina * w_2;
      scalar_t hx = -sina * h_2, hy = cosa * h_2;

      px[1] = roi_x + wx + hx;
      py[1] = roi_y + wy + hy;
      px[2] = roi_x - wx + hx;
      py[2] = roi_y - wy + hy;
      px[3] = roi_x - wx - hx;
      py[3] = roi_y - wy - hy;
      px[4] = roi_x + wx - hx;
      py[4] = roi_y + wy - hy;
    }

    scalar_t* offset_bottom_diff =
        bottom_diff + (n * channels + c) * height * width;
    scalar_t value_top_diff = top_diff[index];

    atomicAdd(bottom_diff + index, value_top_diff);
    for (int i = 0; i < points; i++) {
      scalar_t w1, w2, w3, w4;
      int x_low, x_high, y_low, y_high;

      bilinear_interpolate_gradient<scalar_t>(height, width, py[i], px[i], w1,
                                              w2, w3, w4, x_low, x_high, y_low,
                                              y_high, i);
      scalar_t g1 = value_top_diff * w1;
      scalar_t g2 = value_top_diff * w2;
      scalar_t g3 = value_top_diff * w3;
      scalar_t g4 = value_top_diff * w4;
      if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
        atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
        atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
        atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
        atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
      }
    }
  }
}
#endif  // ROTATED_FEATURE_ALIGN_CUDA_KERNEL_CUH

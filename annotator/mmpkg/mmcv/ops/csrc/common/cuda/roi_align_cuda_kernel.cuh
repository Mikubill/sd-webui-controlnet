// Copyright (c) OpenMMLab. All rights reserved
#ifndef ROI_ALIGN_CUDA_KERNEL_CUH
#define ROI_ALIGN_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT

/*** Forward ***/
template <typename T>
__global__ void roi_align_forward_cuda_kernel(
    const int nthreads, const T* input, const T* rois, T* output, T* argmax_y,
    T* argmax_x, const int pooled_height, const int pooled_width,
    const T spatial_scale, const int sampling_ratio,
    const int pool_mode,  // 0 - max pool, 1 - avg pool
    const bool aligned, const int channels, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {  // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_height / pooled_height));
    int roi_bin_grid_w =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_width / pooled_width));

    if (pool_mode == 0) {
      // We do max pooling inside a bin
      T maxval = -FLT_MAX;
      T maxidx_y = -1.f, maxidx_x = -1.f;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);
          T val =
              bilinear_interpolate(offset_input, height, width, y, x, index);
          if (val > maxval) {
            maxval = val;
            maxidx_y = y;
            maxidx_x = x;
          }
        }
      }
      output[index] = maxval;
      argmax_y[index] = maxidx_y;
      argmax_x[index] = maxidx_x;
    } else if (pool_mode == 1) {
      // We do average pooling inside a bin
      const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
      T output_val = 0.;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);
          T val =
              bilinear_interpolate(offset_input, height, width, y, x, index);
          output_val += val;
        }
      }
      output[index] = output_val / count;
    }
  }
}

/*** Backward ***/
template <typename T>
__global__ void roi_align_backward_cuda_kernel(
    const int nthreads, const T* grad_output, const T* rois, const T* argmax_y,
    const T* argmax_x, T* grad_input, const int pooled_height,
    const int pooled_width, const T spatial_scale, const int sampling_ratio,
    const int pool_mode,  // 0 - max pool, 1 - avg pool
    const bool aligned, const int channels, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T grad_output_this_bin = grad_output[index];

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    T* offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    if (pool_mode == 0) {
      T y = argmax_y[index], x = argmax_x[index];
      if (y != -1.f) {
        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_grad_input + y_low * width + x_low,
                    grad_output_this_bin * w1);
          atomicAdd(offset_grad_input + y_low * width + x_high,
                    grad_output_this_bin * w2);
          atomicAdd(offset_grad_input + y_high * width + x_low,
                    grad_output_this_bin * w3);
          atomicAdd(offset_grad_input + y_high * width + x_high,
                    grad_output_this_bin * w4);
        }
      }
    } else if (pool_mode == 1) {
      // Do not using rounding; this implementation detail is critical
      T offset = aligned ? (T)0.5 : (T)0.0;
      T roi_start_w = offset_rois[1] * spatial_scale - offset;
      T roi_start_h = offset_rois[2] * spatial_scale - offset;
      T roi_end_w = offset_rois[3] * spatial_scale - offset;
      T roi_end_h = offset_rois[4] * spatial_scale - offset;

      T roi_width = roi_end_w - roi_start_w;
      T roi_height = roi_end_h - roi_start_h;
      if (!aligned) {  // for backward-compatibility only
        roi_width = max(roi_width, (T)1.);
        roi_height = max(roi_height, (T)1.);
      }

      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h =
          (sampling_ratio > 0)
              ? sampling_ratio
              : static_cast<int>(ceilf(roi_height / pooled_height));
      int roi_bin_grid_w =
          (sampling_ratio > 0)
              ? sampling_ratio
              : static_cast<int>(ceilf(roi_width / pooled_width));

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);

          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;
          bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                        x_low, x_high, y_low, y_high, index);

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            atomicAdd(offset_grad_input + y_low * width + x_low,
                      grad_output_this_bin * w1 / count);
            atomicAdd(offset_grad_input + y_low * width + x_high,
                      grad_output_this_bin * w2 / count);
            atomicAdd(offset_grad_input + y_high * width + x_low,
                      grad_output_this_bin * w3 / count);
            atomicAdd(offset_grad_input + y_high * width + x_high,
                      grad_output_this_bin * w4 / count);
          }
        }
      }
    }
  }
}

#endif  // ROI_ALIGN_CUDA_KERNEL_CUH

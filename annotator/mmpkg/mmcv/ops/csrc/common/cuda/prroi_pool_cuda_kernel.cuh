// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/vacancy/PreciseRoIPooling/blob/master/src/prroi_pooling_gpu_impl.cu
// Distributed under terms of the MIT license.
#ifndef PRROI_POOL_CUDA_KERNEL_CUH
#define PRROI_POOL_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__device__ static __forceinline__ T PrRoIPoolingGetData(const T *data,
                                                        const int h,
                                                        const int w,
                                                        const int height,
                                                        const int width) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  T retVal = overflow ? 0.0f : data[h * width + w];
  return retVal;
}

template <typename T>
__device__ static __forceinline__ T PrRoIPoolingGetCoeff(T dh, T dw) {
  return (1.0f - abs(dh)) * (1.0f - abs(dw));
}

template <typename T>
__device__ static __forceinline__ T PrRoIPoolingSingleCoorIntegral(T s, T t,
                                                                   T c1, T c2) {
  return 0.5 * (t * t - s * s) * (c2 - c1) + (t - s) * c1;
}

template <typename T>
__device__ static T PrRoIPoolingInterpolation(const T *data, const T h,
                                              const T w, const int height,
                                              const int width) {
  T retVal = 0.0f;
  int h1 = floorf(h);
  int w1 = floorf(w);
  retVal += PrRoIPoolingGetData(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff(h - T(h1), w - T(w1));
  h1 = floorf(h) + 1;
  w1 = floorf(w);
  retVal += PrRoIPoolingGetData(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff(h - T(h1), w - T(w1));
  h1 = floorf(h);
  w1 = floorf(w) + 1;
  retVal += PrRoIPoolingGetData(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff(h - T(h1), w - T(w1));
  h1 = floorf(h) + 1;
  w1 = floorf(w) + 1;
  retVal += PrRoIPoolingGetData(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff(h - T(h1), w - T(w1));
  return retVal;
}

template <typename T>
__device__ static T PrRoIPoolingMatCalculation(const T *this_data,
                                               const int s_h, const int s_w,
                                               const int e_h, const int e_w,
                                               const T y0, const T x0,
                                               const T y1, const T x1,
                                               const int h0, const int w0) {
  T alpha, beta, lim_alpha, lim_beta, tmp;
  T sum_out = 0;

  alpha = x0 - T(s_w);
  beta = y0 - T(s_h);
  lim_alpha = x1 - T(s_w);
  lim_beta = y1 - T(s_h);
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, s_h, s_w, h0, w0) * tmp;

  alpha = T(e_w) - x1;
  lim_alpha = T(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, s_h, e_w, h0, w0) * tmp;

  alpha = x0 - T(s_w);
  beta = T(e_h) - y1;
  lim_alpha = x1 - T(s_w);
  lim_beta = T(e_h) - y0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, e_h, s_w, h0, w0) * tmp;

  alpha = T(e_w) - x1;
  lim_alpha = T(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  sum_out += PrRoIPoolingGetData(this_data, e_h, e_w, h0, w0) * tmp;

  return sum_out;
}

template <typename T>
__device__ static void PrRoIPoolingDistributeDiff(T *diff, const T top_diff,
                                                  const int h, const int w,
                                                  const int height,
                                                  const int width,
                                                  const T coeff) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  if (!overflow) atomicAdd(diff + h * width + w, top_diff * coeff);
}

template <typename T>
__device__ static void PrRoIPoolingMatDistributeDiff(
    T *diff, const T top_diff, const int s_h, const int s_w, const int e_h,
    const int e_w, const T y0, const T x0, const T y1, const T x1, const int h0,
    const int w0) {
  T alpha, beta, lim_alpha, lim_beta, tmp;

  alpha = x0 - T(s_w);
  beta = y0 - T(s_h);
  lim_alpha = x1 - T(s_w);
  lim_beta = y1 - T(s_h);
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  PrRoIPoolingDistributeDiff(diff, top_diff, s_h, s_w, h0, w0, tmp);

  alpha = T(e_w) - x1;
  lim_alpha = T(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  PrRoIPoolingDistributeDiff(diff, top_diff, s_h, e_w, h0, w0, tmp);

  alpha = x0 - T(s_w);
  beta = T(e_h) - y1;
  lim_alpha = x1 - T(s_w);
  lim_beta = T(e_h) - y0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  PrRoIPoolingDistributeDiff(diff, top_diff, e_h, s_w, h0, w0, tmp);

  alpha = T(e_w) - x1;
  lim_alpha = T(e_w) - x0;
  tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha +
         0.5f * alpha * alpha) *
        (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
  PrRoIPoolingDistributeDiff(diff, top_diff, e_h, e_w, h0, w0, tmp);
}

template <typename T>
__global__ void prroi_pool_forward_cuda_kernel(
    const int nthreads, const T *input, const T *rois, T *output,
    const int pooled_height, const int pooled_width, const T spatial_scale,
    const int channels, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T *offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    T roi_x1 = offset_rois[1] * spatial_scale;
    T roi_y1 = offset_rois[2] * spatial_scale;
    T roi_x2 = offset_rois[3] * spatial_scale;
    T roi_y2 = offset_rois[4] * spatial_scale;

    T roi_width = max(roi_x2 - roi_x1, ((T)0.0));
    T roi_height = max(roi_y2 - roi_y1, ((T)0.0));
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    const T *this_data =
        input + (roi_batch_ind * channels + c) * height * width;
    T *this_out = output + index;

    T bin_x1 = roi_x1 + bin_size_w * pw;
    T bin_y1 = roi_y1 + bin_size_h * ph;
    T bin_x2 = bin_x1 + bin_size_w;
    T bin_y2 = bin_y1 + bin_size_h;

    T bin_size = max(T(0.0), bin_size_w * bin_size_h);
    if (bin_size == 0) {
      *this_out = 0;
      continue;
    }

    T sum_out = 0;

    int start_x, start_y, end_x, end_y;

    start_x = floorf(bin_x1);
    end_x = ceilf(bin_x2);
    start_y = floorf(bin_y1);
    end_y = ceilf(bin_y2);

    for (int bin_x = start_x; bin_x < end_x; ++bin_x)
      for (int bin_y = start_y; bin_y < end_y; ++bin_y)
        sum_out += PrRoIPoolingMatCalculation(
            this_data, bin_y, bin_x, bin_y + 1, bin_x + 1,
            max(bin_y1, T(bin_y)), max(bin_x1, T(bin_x)),
            min(bin_y2, T(bin_y) + 1.0f), min(bin_x2, T(bin_x + 1.0f)), height,
            width);
    *this_out = sum_out / bin_size;
  }
}

template <typename T>
__global__ void prroi_pool_backward_cuda_kernel(
    const int nthreads, const T *grad_output, const T *rois, T *grad_input,
    const int pooled_height, const int pooled_width, const T spatial_scale,
    const int channels, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    auto rois_cur = rois + n * 5;

    int roi_batch_ind = rois_cur[0];
    T roi_x1 = rois_cur[1] * spatial_scale;
    T roi_y1 = rois_cur[2] * spatial_scale;
    T roi_x2 = rois_cur[3] * spatial_scale;
    T roi_y2 = rois_cur[4] * spatial_scale;

    T roi_width = max(roi_x2 - roi_x1, (T)0);
    T roi_height = max(roi_y2 - roi_y1, (T)0);
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    const T *this_out_grad = grad_output + index;
    T *this_data_grad =
        grad_input + (roi_batch_ind * channels + c) * height * width;

    T bin_x1 = roi_x1 + bin_size_w * pw;
    T bin_y1 = roi_y1 + bin_size_h * ph;
    T bin_x2 = bin_x1 + bin_size_w;
    T bin_y2 = bin_y1 + bin_size_h;

    T bin_size = max(T(0.0), bin_size_w * bin_size_h);

    T sum_out = bin_size == T(0) ? T(0) : *this_out_grad / bin_size;

    int start_x, start_y, end_x, end_y;

    start_x = floorf(bin_x1);
    end_x = ceilf(bin_x2);
    start_y = floorf(bin_y1);
    end_y = ceilf(bin_y2);

    for (int bin_x = start_x; bin_x < end_x; ++bin_x)
      for (int bin_y = start_y; bin_y < end_y; ++bin_y)
        PrRoIPoolingMatDistributeDiff(
            this_data_grad, sum_out, bin_y, bin_x, bin_y + 1, bin_x + 1,
            max(bin_y1, T(bin_y)), max(bin_x1, T(bin_x)),
            min(bin_y2, T(bin_y) + 1.0f), min(bin_x2, T(bin_x + 1.0f)), height,
            width);
  }
}

template <typename T>
__global__ void prroi_pool_coor_backward_cuda_kernel(
    const int nthreads, const T *output, const T *grad_output, const T *input,
    const T *rois, T *grad_rois, const int pooled_height,
    const int pooled_width, const T spatial_scale, const int channels,
    const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    auto rois_cur = rois + n * 5;

    int roi_batch_ind = rois_cur[0];
    T roi_x1 = rois_cur[1] * spatial_scale;
    T roi_y1 = rois_cur[2] * spatial_scale;
    T roi_x2 = rois_cur[3] * spatial_scale;
    T roi_y2 = rois_cur[4] * spatial_scale;

    T roi_width = max(roi_x2 - roi_x1, (T)0);
    T roi_height = max(roi_y2 - roi_y1, (T)0);
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    const T output_grad_val = grad_output[index];
    const T *this_input_data =
        input + (roi_batch_ind * channels + c) * height * width;
    const T output_val = output[index];
    T *this_rois_grad = grad_rois + n * 5;

    T bin_x1 = roi_x1 + bin_size_w * pw;
    T bin_y1 = roi_y1 + bin_size_h * ph;
    T bin_x2 = bin_x1 + bin_size_w;
    T bin_y2 = bin_y1 + bin_size_h;

    T bin_size = max(T(0.0), bin_size_w * bin_size_h);

    T sum_out = bin_size == T(0) ? T(0) : output_grad_val / bin_size;

    // WARNING: to be discussed
    if (sum_out == 0) continue;

    int start_x, start_y, end_x, end_y;

    start_x = floorf(bin_x1);
    end_x = ceilf(bin_x2);
    start_y = floorf(bin_y1);
    end_y = ceilf(bin_y2);

    T grad_x1_y = 0, grad_x2_y = 0, grad_x_y1 = 0, grad_x_y2 = 0;
    for (int bin_y = start_y; bin_y < end_y; ++bin_y) {
      grad_x1_y += PrRoIPoolingSingleCoorIntegral(
          max(bin_y1, T(bin_y)) - bin_y, min(bin_y2, T(bin_y + 1)) - bin_y,
          PrRoIPoolingInterpolation(this_input_data, float(bin_y), bin_x1,
                                    height, width),
          PrRoIPoolingInterpolation(this_input_data, float(bin_y + 1), bin_x1,
                                    height, width));

      grad_x2_y += PrRoIPoolingSingleCoorIntegral(
          max(bin_y1, T(bin_y)) - bin_y, min(bin_y2, T(bin_y + 1)) - bin_y,
          PrRoIPoolingInterpolation(this_input_data, float(bin_y), bin_x2,
                                    height, width),
          PrRoIPoolingInterpolation(this_input_data, float(bin_y + 1), bin_x2,
                                    height, width));
    }

    for (int bin_x = start_x; bin_x < end_x; ++bin_x) {
      grad_x_y1 += PrRoIPoolingSingleCoorIntegral(
          max(bin_x1, T(bin_x)) - bin_x, min(bin_x2, T(bin_x + 1)) - bin_x,
          PrRoIPoolingInterpolation(this_input_data, bin_y1, float(bin_x),
                                    height, width),
          PrRoIPoolingInterpolation(this_input_data, bin_y1, float(bin_x + 1),
                                    height, width));

      grad_x_y2 += PrRoIPoolingSingleCoorIntegral(
          max(bin_x1, T(bin_x)) - bin_x, min(bin_x2, T(bin_x + 1)) - bin_x,
          PrRoIPoolingInterpolation(this_input_data, bin_y2, float(bin_x),
                                    height, width),
          PrRoIPoolingInterpolation(this_input_data, bin_y2, float(bin_x + 1),
                                    height, width));
    }

    T partial_x1 = -grad_x1_y + (bin_y2 - bin_y1) * output_val;
    T partial_y1 = -grad_x_y1 + (bin_x2 - bin_x1) * output_val;
    T partial_x2 = grad_x2_y - (bin_y2 - bin_y1) * output_val;
    T partial_y2 = grad_x_y2 - (bin_x2 - bin_x1) * output_val;

    partial_x1 = partial_x1 / bin_size * spatial_scale;
    partial_x2 = partial_x2 / bin_size * spatial_scale;
    partial_y1 = partial_y1 / bin_size * spatial_scale;
    partial_y2 = partial_y2 / bin_size * spatial_scale;

    // (index, x1, y1, x2, y2)
    this_rois_grad[0] = 0;
    atomicAdd(this_rois_grad + 1,
              (partial_x1 * (1.0f - T(pw) / pooled_width) +
               partial_x2 * (1.0f - T(pw + 1) / pooled_width)) *
                  output_grad_val);
    atomicAdd(this_rois_grad + 2,
              (partial_y1 * (1.0f - T(ph) / pooled_height) +
               partial_y2 * (1.0f - T(ph + 1) / pooled_height)) *
                  output_grad_val);
    atomicAdd(this_rois_grad + 3, (partial_x2 * T(pw + 1) / pooled_width +
                                   partial_x1 * T(pw) / pooled_width) *
                                      output_grad_val);
    atomicAdd(this_rois_grad + 4, (partial_y2 * T(ph + 1) / pooled_height +
                                   partial_y1 * T(ph) / pooled_height) *
                                      output_grad_val);
  }
}

#endif  // ROI_POOL_CUDA_KERNEL_CUH

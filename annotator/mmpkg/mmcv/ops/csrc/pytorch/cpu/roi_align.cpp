// Modified from
// https://github.com/facebookresearch/detectron2/tree/master/detectron2/layers/csrc/ROIAlign
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

// implementation taken from Caffe2
template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int iy_upper, const int ix_upper,
    T roi_start_h, T roi_start_w, T bin_size_h, T bin_size_w,
    int roi_bin_grid_h, int roi_bin_grid_w, std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
                     static_cast<T>(iy + .5f) * bin_size_h /
                         static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
                       static_cast<T>(ix + .5f) * bin_size_w /
                           static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indices
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward(const int nthreads, const T* input, const T* rois,
                     T* output, T* argmax_y, T* argmax_x,
                     const int pooled_height, const int pooled_width,
                     const T spatial_scale, const int sampling_ratio,
                     const int pool_mode,  // 0 - max pool, 1 - avg pool
                     const bool aligned, const int channels, const int height,
                     const int width) {
  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (aligned) {
      AT_ASSERTM(roi_width >= 0 && roi_height >= 0,
                 "ROIs in ROIAlign cannot have non-negative size!");
    } else {  // for backward-compatibility only
      roi_width = std::max(roi_width, (T)1.);
      roi_height = std::max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceilf(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooled_width);

    // When the grid is empty, output zeros == 0/1, instead of NaN.
    const T count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    // we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w *
                                     pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height, width, pooled_height, pooled_width, roi_bin_grid_h,
        roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w,
        roi_bin_grid_h, roi_bin_grid_w, pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const T* offset_input =
          input + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          T maxval = -10000;
          T maxidx_y = -1.f, maxidx_x = -1.f;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = roi_start_h + ph * bin_size_h +
                        static_cast<T>(iy + .5f) * bin_size_h /
                            static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              const T x = roi_start_w + pw * bin_size_w +
                          static_cast<T>(ix + .5f) * bin_size_w /
                              static_cast<T>(roi_bin_grid_w);
              PreCalc<T> pc = pre_calc[pre_calc_index];
              T val = pc.w1 * offset_input[pc.pos1] +
                      pc.w2 * offset_input[pc.pos2] +
                      pc.w3 * offset_input[pc.pos3] +
                      pc.w4 * offset_input[pc.pos4];
              if (val > maxval) {
                maxval = val;
                maxidx_y = y;
                maxidx_x = x;
              }
              output_val += val;
              pre_calc_index += 1;
            }
          }
          if (pool_mode == 0) {
            // We do max pooling inside a bin
            output[index] = maxval;
            argmax_y[index] = maxidx_y;
            argmax_x[index] = maxidx_x;
          } else if (pool_mode == 1) {
            // We do average (integral) pooling inside a bin
            output[index] = output_val / count;
          }  // if
        }    // for pw
      }      // for ph
    }        // for c
  }          // for n
}

template <typename T>
void bilinear_interpolate_gradient(const int height, const int width, T y, T x,
                                   T& w1, T& w2, T& w3, T& w4, int& x_low,
                                   int& x_high, int& y_low, int& y_high,
                                   const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <class T>
inline void add(T* address, const T& val) {
  *address += val;
}

template <typename T>
void ROIAlignBackward(const int nthreads, const T* grad_output, const T* rois,
                      const T* argmax_y, const T* argmax_x, T* grad_input,
                      const int pooled_height, const int pooled_width,
                      const T spatial_scale, const int sampling_ratio,
                      const int pool_mode,  // 0 - max pool, 1 - avg pool
                      const bool aligned, const int channels, const int height,
                      const int width, const int n_stride, const int c_stride,
                      const int h_stride, const int w_stride) {
  for (int index = 0; index < nthreads; index++) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (aligned) {
      AT_ASSERTM(roi_width >= 0 && roi_height >= 0,
                 "ROIs in ROIAlign do not have non-negative size!");
    } else {  // for backward-compatibility only
      roi_width = std::max(roi_width, (T)1.);
      roi_height = std::max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    int output_offset = n * n_stride + c * c_stride;
    const T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride];

    if (pool_mode == 0) {
      // We do max pooling inside a bin
      T y = argmax_y[index], x = argmax_x[index];
      if (y != -1.f) {
        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);

        T g1 = grad_output_this_bin * w1;
        T g2 = grad_output_this_bin * w2;
        T g3 = grad_output_this_bin * w3;
        T g4 = grad_output_this_bin * w4;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          // atomic add is not needed for now since it is single threaded
          add(offset_grad_input + y_low * width + x_low, static_cast<T>(g1));
          add(offset_grad_input + y_low * width + x_high, static_cast<T>(g2));
          add(offset_grad_input + y_high * width + x_low, static_cast<T>(g3));
          add(offset_grad_input + y_high * width + x_high, static_cast<T>(g4));
        }  // if
      }    // mode
    } else if (pool_mode == 1) {
      // We do average (integral) pooling inside a bin
      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h =
          (sampling_ratio > 0)
              ? sampling_ratio
              : ceilf(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0)
                               ? sampling_ratio
                               : ceilf(roi_width / pooled_width);

      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
                      static_cast<T>(ix + .5f) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);

          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;

          bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,
                                        x_low, x_high, y_low, y_high, index);

          T g1 = grad_output_this_bin * w1 / count;
          T g2 = grad_output_this_bin * w2 / count;
          T g3 = grad_output_this_bin * w3 / count;
          T g4 = grad_output_this_bin * w4 / count;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            // atomic add is not needed for now since it is single threaded
            add(offset_grad_input + y_low * width + x_low, static_cast<T>(g1));
            add(offset_grad_input + y_low * width + x_high, static_cast<T>(g2));
            add(offset_grad_input + y_high * width + x_low, static_cast<T>(g3));
            add(offset_grad_input + y_high * width + x_high,
                static_cast<T>(g4));
          }  // if
        }    // ix
      }      // iy
    }        // mode
  }          // for
}  // ROIAlignBackward

void ROIAlignForwardCPULauncher(Tensor input, Tensor rois, Tensor output,
                                Tensor argmax_y, Tensor argmax_x,
                                int aligned_height, int aligned_width,
                                float spatial_scale, int sampling_ratio,
                                int pool_mode, bool aligned) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ROIAlign_forward", [&] {
        ROIAlignForward<scalar_t>(
            output_size, input.data_ptr<scalar_t>(), rois.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
            argmax_x.data_ptr<scalar_t>(), aligned_height, aligned_width,
            static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
            aligned, channels, height, width);
      });
}

void ROIAlignBackwardCPULauncher(Tensor grad_output, Tensor rois,
                                 Tensor argmax_y, Tensor argmax_x,
                                 Tensor grad_input, int aligned_height,
                                 int aligned_width, float spatial_scale,
                                 int sampling_ratio, int pool_mode,
                                 bool aligned) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  // get stride values to ensure indexing into gradients is correct.
  int n_stride = grad_output.stride(0);
  int c_stride = grad_output.stride(1);
  int h_stride = grad_output.stride(2);
  int w_stride = grad_output.stride(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "ROIAlign_backward", [&] {
        ROIAlignBackward<scalar_t>(
            output_size, grad_output.data_ptr<scalar_t>(),
            rois.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
            argmax_x.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
            aligned_height, aligned_width, static_cast<scalar_t>(spatial_scale),
            sampling_ratio, pool_mode, aligned, channels, height, width,
            n_stride, c_stride, h_stride, w_stride);
      });
}

void roi_align_forward_cpu(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax_y, Tensor argmax_x, int aligned_height,
                           int aligned_width, float spatial_scale,
                           int sampling_ratio, int pool_mode, bool aligned) {
  ROIAlignForwardCPULauncher(input, rois, output, argmax_y, argmax_x,
                             aligned_height, aligned_width, spatial_scale,
                             sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_cpu(Tensor grad_output, Tensor rois, Tensor argmax_y,
                            Tensor argmax_x, Tensor grad_input,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  ROIAlignBackwardCPULauncher(grad_output, rois, argmax_y, argmax_x, grad_input,
                              aligned_height, aligned_width, spatial_scale,
                              sampling_ratio, pool_mode, aligned);
}

void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned);

REGISTER_DEVICE_IMPL(roi_align_forward_impl, CPU, roi_align_forward_cpu);
REGISTER_DEVICE_IMPL(roi_align_backward_impl, CPU, roi_align_backward_cpu);

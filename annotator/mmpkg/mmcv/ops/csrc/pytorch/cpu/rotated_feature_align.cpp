// modified from
// https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection/blob/master/mmdet/ops/fr/src/feature_refine_kernel.cu
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

template <typename T>
T bilinear_interpolate(const T* input, const int height, const int width, T y,
                       T x, const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

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
  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  const T v_low = fma(v2 - v1, lx, v1);
  const T v_high = fma(v4 - v3, lx, v3);
  const T val = fma(v_high - v_low, ly, v_low);

  return val;
}

template <typename scalar_t>
void rotated_feature_align_forward_cpu_kernel(
    const int nthreads, const int points, const scalar_t* bottom_data,
    const scalar_t* best_bboxes, const scalar_t spatial_scale,
    const int channels, const int height, const int width, scalar_t* top_data) {
  for (int index = 0; index < nthreads; index++) {
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

template <typename T>
void bilinear_interpolate_gradient(const int height, const int width, T y, T x,
                                   T& w1, T& w2, T& w3, T& w4, int& x_low,
                                   int& x_high, int& y_low, int& y_high,
                                   const int index) {
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

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
inline void valueAdd(scalar_t* address, scalar_t val) {
  scalar_t old = *address;
  *address = (old + val);
}

template <typename scalar_t>
void rotated_feature_align_backward_cpu_kernel(
    const int nthreads, const int points, const scalar_t* top_diff,
    const scalar_t* best_bboxes, const scalar_t spatial_scale,
    const int channels, const int height, const int width,
    scalar_t* bottom_diff) {
  for (int index = 0; index < nthreads; index++) {
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

    valueAdd(bottom_diff + index, value_top_diff);
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
        valueAdd(offset_bottom_diff + y_low * width + x_low, g1);
        valueAdd(offset_bottom_diff + y_low * width + x_high, g2);
        valueAdd(offset_bottom_diff + y_high * width + x_low, g3);
        valueAdd(offset_bottom_diff + y_high * width + x_high, g4);
      }
    }
  }
}

void rotated_feature_align_forward_cpu(const Tensor features,
                                       const Tensor best_bboxes,
                                       const float spatial_scale,
                                       const int points, Tensor output) {
  const int output_size = features.numel();
  AT_DISPATCH_FLOATING_TYPES(
      features.scalar_type(), "rotated_feature_align_forward_cpu_kernel", [&] {
        const scalar_t* bottom_data = features.data_ptr<scalar_t>();
        const scalar_t* bboxes_data = best_bboxes.data_ptr<scalar_t>();
        scalar_t* top_data = output.data_ptr<scalar_t>();

        rotated_feature_align_forward_cpu_kernel<scalar_t>(
            output_size, points, bottom_data, bboxes_data,
            scalar_t(spatial_scale), features.size(1), features.size(2),
            features.size(3), top_data);
      });
}

void rotated_feature_align_backward_cpu(const Tensor top_grad,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor bottom_grad) {
  const int output_size = top_grad.numel();
  AT_DISPATCH_FLOATING_TYPES(
      top_grad.scalar_type(), "rotated_feature_align_backward_cpu_kernel", [&] {
        const scalar_t* top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t* bboxes_data = best_bboxes.data_ptr<scalar_t>();
        scalar_t* bottom_diff = bottom_grad.data_ptr<scalar_t>();

        rotated_feature_align_backward_cpu_kernel<scalar_t>(
            output_size, points, top_diff, bboxes_data, scalar_t(spatial_scale),
            top_grad.size(1), top_grad.size(2), top_grad.size(3), bottom_diff);
      });
}

void rotated_feature_align_forward_impl(const Tensor features,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor output);

void rotated_feature_align_backward_impl(const Tensor top_grad,
                                         const Tensor best_bboxes,
                                         const float spatial_scale,
                                         const int points, Tensor bottom_grad);

REGISTER_DEVICE_IMPL(rotated_feature_align_forward_impl, CPU,
                     rotated_feature_align_forward_cpu);

REGISTER_DEVICE_IMPL(rotated_feature_align_backward_impl, CPU,
                     rotated_feature_align_backward_cpu);

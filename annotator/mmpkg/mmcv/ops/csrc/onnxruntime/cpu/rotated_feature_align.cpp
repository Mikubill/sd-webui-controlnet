// Modified from
// https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection/blob/master/mmdet/ops/fr/src/feature_refine_kernel.cu
#include "rotated_feature_align.h"

#include "../ort_mmcv_utils.h"

template <typename T>
T bilinear_interpolate(const T *input, const int height, const int width, T y,
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
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = input[int(fma(y_low, width, x_low))];
  T v2 = input[int(fma(y_low, width, x_high))];
  T v3 = input[int(fma(y_high, width, x_low))];
  T v4 = input[int(fma(y_high, width, x_high))];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename scalar_t>
void rotated_feature_align_forward_cpu_kernel(
    const int nthreads, const int points, const scalar_t *bottom_data,
    const scalar_t *best_bboxes, const scalar_t spatial_scale,
    const int channels, const int height, const int width, scalar_t *top_data) {
  for (int index = 0; index < nthreads; index++) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    const scalar_t *bbox_offset =
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

    const scalar_t *offset_bottom_data =
        bottom_data + (n * channels + c) * height * width;

    scalar_t output_val = bottom_data[index];
    for (int i = 0; i < points; i++) {
      output_val += bilinear_interpolate<scalar_t>(offset_bottom_data, height,
                                                   width, py[i], px[i], i);
    }
    top_data[index] = output_val;
  }
}

void MMCVRotatedFeatureAlignKernel::Compute(OrtKernelContext *context) {
  // Setup inputs
  const OrtValue *input_features = ort_.KernelContext_GetInput(context, 0);
  const float *features_data = reinterpret_cast<const float *>(
      ort_.GetTensorData<float>(input_features));
  const OrtValue *input_best_rbboxes = ort_.KernelContext_GetInput(context, 1);
  const float *best_rbboxes = reinterpret_cast<const float *>(
      ort_.GetTensorData<const float *>(input_best_rbboxes));

  // Setup output
  OrtTensorDimensions out_dimensions(ort_, input_features);

  int batch_size = out_dimensions.data()[0];
  int input_channels = out_dimensions.data()[1];
  int input_height = out_dimensions.data()[2];
  int input_width = out_dimensions.data()[3];

  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, out_dimensions.data(), out_dimensions.size());
  float *out = ort_.GetTensorMutableData<float>(output);
  OrtTensorTypeAndShapeInfo *output_info = ort_.GetTensorTypeAndShape(output);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // TODO: forward here
  int output_size = out_dimensions.data()[0];
  for (auto i = 1; i < out_dimensions.size(); ++i) {
    output_size *= out_dimensions.data()[i];
  }
  rotated_feature_align_forward_cpu_kernel<float>(
      output_size, points_, features_data, best_rbboxes, spatial_scale_,
      input_channels, input_height, input_width, out);
}

// Copyright (c) OpenMMLab. All rights reserved
#include "deform_conv.h"

#include <cmath>
#include <vector>

#include "../ort_mmcv_utils.h"

void gemm_ref_fp32_deform(const float *A, const float *B, const float *V,
                          const float *H, const int32_t trans_A,
                          const int32_t trans_B, const int32_t M,
                          const int32_t N, const int32_t K, const float alpha,
                          const float beta, float *Y) {
  if (!trans_A && !trans_B) {  // MK, KN; NN
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float y = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          y += A[m * K + k] * B[k * N + n];
        }
        y *= alpha;
        if (V) y += beta * V[n];
        if (H) y += beta * H[m * N + n];
        Y[m * N + n] = y;
      }
    }
  }
  if (trans_A && !trans_B) {  // KM, KN; TN
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float y = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          y += A[k * M + m] * B[k * N + n];
        }
        y *= alpha;
        if (V) y += beta * V[n];
        if (H) y += beta * H[m * N + n];
        Y[m * N + n] = y;
      }
    }
  }
  if (trans_A && trans_B) {  // KM, NK; TT
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float y = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          y += A[k * M + m] * B[n * K + k];
        }
        y *= alpha;
        if (V) y += beta * V[n];
        if (H) y += beta * H[m * N + n];
        Y[m * N + n] = y;
      }
    }
  }
  if (!trans_A && trans_B) {  // MK, NK; NT
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float y = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          y += A[m * K + k] * B[n * K + k];
        }
        y *= alpha;
        if (V) y += beta * V[n];
        if (H) y += beta * H[m * N + n];
        Y[m * N + n] = y;
      }
    }
  }
}

float bilinear_interpolate(const float *src, const int64_t src_h,
                           const int64_t src_w, const float h, const float w) {
  if (h <= -1 || src_h <= h || w <= -1 || src_w <= w) {
    return 0;
  }

  int64_t h_low = floor(h);
  int64_t w_low = floor(w);
  int64_t h_high = h_low + 1;
  int64_t w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh;
  float hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = src[h_low * src_w + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= src_w - 1) v2 = src[h_low * src_w + w_high];
  float v3 = 0;
  if (h_high <= src_h - 1 && w_low >= 0) v3 = src[h_high * src_w + w_low];
  float v4 = 0;
  if (h_high <= src_h - 1 && w_high <= src_w - 1)
    v4 = src[h_high * src_w + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

void deformable_im2col(const float *input, const float *offset,
                       const int64_t src_h, const int64_t src_w,
                       const int64_t kernel_h, const int64_t kernel_w,
                       const int64_t pad_h, const int64_t pad_w,
                       const int64_t stride_h, const int64_t stride_w,
                       const int64_t dilation_h, const int64_t dilation_w,
                       const int64_t channels, const int64_t offset_groups,
                       const int64_t dst_h, const int64_t dst_w,
                       float *columns) {
  const int64_t indices = channels * dst_h * dst_w;
  for (int64_t index = 0; index != indices; ++index) {
    const int64_t w_col = index % dst_w;
    const int64_t h_col = (index / dst_w) % dst_h;
    const int64_t c_im = index / (dst_w * dst_h);
    const int64_t c_col = c_im * kernel_h * kernel_w;

    int64_t c_per_offset_grp = channels / offset_groups;
    const int64_t grp_idx = c_im / c_per_offset_grp;
    auto columns_ptr =
        columns + (c_col * (dst_h * dst_w) + h_col * dst_w + w_col);
    auto input_ptr = input + c_im * (src_h * src_w);
    auto offset_ptr =
        offset + grp_idx * 2 * kernel_h * kernel_w * dst_h * dst_w;

    for (int64_t kh = 0; kh < kernel_h; ++kh) {
      for (int64_t kw = 0; kw < kernel_w; ++kw) {
        const int data_offset_h_ptr =
            ((2 * (kh * kernel_w + kw)) * dst_h + h_col) * dst_w + w_col;
        const int data_offset_w_ptr =
            ((2 * (kh * kernel_w + kw) + 1) * dst_h + h_col) * dst_w + w_col;

        const float offset_h = offset_ptr[data_offset_h_ptr];
        const float offset_w = offset_ptr[data_offset_w_ptr];
        const float ih =
            (h_col * stride_h - pad_h) + kh * dilation_h + offset_h;
        const float iw =
            (w_col * stride_w - pad_w) + kw * dilation_w + offset_w;
        *columns_ptr = bilinear_interpolate(input_ptr, src_h, src_w, ih, iw);
        columns_ptr += dst_h * dst_w;
      }
    }
  }
}

void deformable_conv_forward(
    const float *src, const float *offset, const float *filter,
    const int64_t batch, const int64_t src_c, const int64_t src_h,
    const int64_t src_w, const int64_t dst_c, const int64_t dst_h,
    const int64_t dst_w, const int64_t group, const int64_t offset_group,
    const int64_t channels, const int64_t num_output, const int64_t kernel_h,
    const int64_t kernel_w, const int64_t stride_h, const int64_t stride_w,
    const int64_t pad_h, const int64_t pad_w, const int64_t dilation_h,
    const int64_t dilation_w, float *columns, float *dst) {
  const int64_t ic_per_gp = channels / group;
  const int64_t oc_per_gp = num_output / group;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t g = 0; g < group; ++g) {
      deformable_im2col(
          src + b * src_c * src_h * src_w + g * ic_per_gp * src_h * src_w,
          offset + b * offset_group * 2 * kernel_h * kernel_w * dst_h * dst_w,
          src_h, src_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
          dilation_h, dilation_w, ic_per_gp, offset_group, dst_h, dst_w,
          columns);
      float *dst_ptr =
          dst + b * dst_c * dst_h * dst_w + g * oc_per_gp * dst_h * dst_w;

      memset(dst_ptr, 0.0f, sizeof(float) * oc_per_gp * dst_h * dst_w);

      gemm_ref_fp32_deform(
          filter + g * oc_per_gp * ic_per_gp * kernel_h * kernel_w, columns,
          nullptr, dst_ptr, 0, 0, oc_per_gp, dst_h * dst_w,
          ic_per_gp * kernel_h * kernel_w, 1.0f, 1.0f, dst_ptr);
    }
  }
}

MMCVDeformConvKernel::MMCVDeformConvKernel(OrtApi api,
                                           const OrtKernelInfo *info)
    : api_(api), ort_(api_), info_(info) {
  std::vector<int64_t> stride =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "stride");
  stride_height_ = stride[0];
  stride_width_ = stride[1];
  std::vector<int64_t> padding =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "padding");
  padding_height_ = padding[0];
  padding_width_ = padding[1];
  std::vector<int64_t> dilation =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "dilation");
  dilation_height_ = dilation[0];
  dilation_width_ = dilation[1];
  deformable_group_ =
      ort_.KernelInfoGetAttribute<int64_t>(info, "deform_groups");
  group_ = ort_.KernelInfoGetAttribute<int64_t>(info, "groups");

  // create allocator
  allocator_ = Ort::AllocatorWithDefaultOptions();
}

void MMCVDeformConvKernel::Compute(OrtKernelContext *context) {
  const int64_t stride_height = stride_height_;
  const int64_t stride_width = stride_width_;
  const int64_t padding_height = padding_height_;
  const int64_t padding_width = padding_width_;
  const int64_t dilation_height = dilation_height_;
  const int64_t dilation_width = dilation_width_;
  const int64_t deformable_group = deformable_group_;
  const int64_t group = group_;

  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  const float *input_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(input));

  const OrtValue *offset = ort_.KernelContext_GetInput(context, 1);
  const float *offset_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(offset));

  const OrtValue *filter = ort_.KernelContext_GetInput(context, 2);
  const float *filter_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(filter));

  OrtTensorDimensions input_dims(ort_, input);
  OrtTensorDimensions filter_dims(ort_, filter);

  int64_t batch_size = input_dims[0];
  int64_t in_channels = input_dims[1];
  int64_t in_height = input_dims[2];
  int64_t in_width = input_dims[3];
  int64_t out_channels = filter_dims[0];
  int64_t kernel_height = filter_dims[2];
  int64_t kernel_width = filter_dims[3];

  // get output memory
  int64_t out_height = floor((in_height + 2 * padding_height -
                              dilation_height * (kernel_height - 1) - 1) /
                                 stride_height +
                             1);
  int64_t out_width = floor(
      (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) /
          stride_width +
      1);

  std::vector<int64_t> output_dims = {batch_size, out_channels, out_height,
                                      out_width};

  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, output_dims.data(), output_dims.size());
  float *out_ptr = ort_.GetTensorMutableData<float>(output);

  // allocate tmp memory
  int64_t column_len = (in_channels / group) * kernel_height * kernel_width *
                       out_height * out_width;
  float *columns = (float *)allocator_.Alloc(sizeof(float) * column_len);
  deformable_conv_forward(
      input_data, offset_data, filter_data, batch_size, in_channels, in_height,
      in_width, out_channels, out_height, out_width, group, deformable_group,
      in_channels, out_channels, kernel_height, kernel_width, stride_height,
      stride_width, padding_height, padding_width, dilation_height,
      dilation_width, columns, out_ptr);
}

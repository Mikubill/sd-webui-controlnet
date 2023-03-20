// Copyright (c) OpenMMLab. All rights reserved
#include <cuda_fp16.h>

#include "common_cuda_helper.hpp"
#include "deform_conv_cuda_kernel.cuh"
#include "trt_cuda_helper.cuh"
#include "trt_plugin_helper.hpp"

template <typename T>
void trt_deformable_im2col(const T* data_input, const T* data_offset,
                           const int channels, const int height,
                           const int width, const int ksize_h,
                           const int ksize_w, const int pad_h, const int pad_w,
                           const int stride_h, const int stride_w,
                           const int dilation_h, const int dilation_w,
                           const int parallel_imgs, const int deformable_group,
                           T* data_col, cudaStream_t stream) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  deformable_im2col_gpu_kernel<T>
      <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
          num_kernels, data_input, data_offset, height, width, ksize_h, ksize_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          channel_per_deformable_group, parallel_imgs, channels,
          deformable_group, height_col, width_col, data_col);

  cudaCheckError();
}

template <typename scalar_t>
void DeformConvForwardCUDAKernelLauncher(
    const scalar_t* input, const scalar_t* weight, const scalar_t* offset,
    scalar_t* output, void* workspace, int batchSize, int nInputPlane,
    int inputHeight, int inputWidth, int nOutputPlane, int kW, int kH, int dW,
    int dH, int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step, cublasHandle_t cublas_handle,
    cudaStream_t stream) {
  size_t word_size = sizeof(scalar_t);

  im2col_step = std::min(int(batchSize), im2col_step);
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long long columns_size =
      mmcv::getAlignedSize(nInputPlane * kW * kH * im2col_step * outputHeight *
                           outputWidth * word_size);

  // column buffer for img2col
  scalar_t* columns = (scalar_t*)workspace;
  workspace = workspace + columns_size;

  scalar_t* output_buffer;
  long long output_buffer_size = 0;
  if (im2col_step == 1) {
    output_buffer = output;
  } else {
    // output need permute when im2col_step!=1
    output_buffer = (scalar_t*)workspace;
    output_buffer_size = batchSize * nOutputPlane * outputWidth * outputHeight;
  }

  long long input_elt_step =
      im2col_step * nInputPlane * inputHeight * inputWidth;
  long long offset_elt_step =
      im2col_step * deformable_group * 2 * kH * kW * outputHeight * outputWidth;
  long long out_buffer_step =
      nOutputPlane * im2col_step * outputHeight * outputWidth;
  long long col_g_step =
      nInputPlane * kW * kH / group * im2col_step * outputHeight * outputWidth;
  long long weight_g_step =
      nOutputPlane / group * nInputPlane / group * kH * kW;
  long long out_buffer_g_step =
      nOutputPlane / group * im2col_step * outputHeight * outputWidth;
  int m = nOutputPlane / group;
  int n = im2col_step * outputHeight * outputWidth;
  int k = nInputPlane / group * kH * kW;
  scalar_t alpha = 1.;
  scalar_t beta = 0.;

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    const scalar_t* input_start = input + elt * input_elt_step;
    const scalar_t* offset_start = offset + elt * offset_elt_step;

    trt_deformable_im2col<scalar_t>(input_start, offset_start, nInputPlane,
                                    inputHeight, inputWidth, kH, kW, padH, padW,
                                    dH, dW, dilationH, dilationW, im2col_step,
                                    deformable_group, columns, stream);

    for (int g = 0; g < group; ++g) {
      const scalar_t* weight_start = weight + g * weight_g_step;
      scalar_t* col_start = columns + g * col_g_step;
      scalar_t* out_buffer_start =
          output_buffer + elt * out_buffer_step + g * out_buffer_g_step;

      cublasGemmWrap<scalar_t>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                               &alpha, col_start, n, weight_start, k, &beta,
                               out_buffer_start, n);
      cudaCheckError();
    }
  }

  if (im2col_step != 1) {
    int output_buffer_shape[5] = {batchSize / im2col_step, nOutputPlane,
                                  im2col_step, outputHeight, outputWidth};
    int output_buffer_permute[5] = {0, 2, 1, 3, 4};
    memcpyPermute<scalar_t>(output, output_buffer, &output_buffer_shape[0],
                            &output_buffer_permute[0], 5, stream);
  }
}

void DeformConvForwardCUDAKernelLauncher_float(
    const float* input, const float* weight, const float* offset, float* output,
    void* workspace, int batchSize, int nInputPlane, int inputHeight,
    int inputWidth, int nOutputPlane, int kW, int kH, int dW, int dH, int padW,
    int padH, int dilationW, int dilationH, int group, int deformable_group,
    int im2col_step, cublasHandle_t cublas_handle, cudaStream_t stream) {
  DeformConvForwardCUDAKernelLauncher<float>(
      input, weight, offset, output, workspace, batchSize, nInputPlane,
      inputHeight, inputWidth, nOutputPlane, kW, kH, dW, dH, padW, padH,
      dilationW, dilationH, group, deformable_group, im2col_step, cublas_handle,
      stream);
}

// Copyright (c) OpenMMLab. All rights reserved
#include <assert.h>
#include <cuda_fp16.h>

#include "common_cuda_helper.hpp"
#include "modulated_deform_conv_cuda_kernel.cuh"
#include "trt_cuda_helper.cuh"
#include "trt_plugin_helper.hpp"

template <typename T>
void trt_modulated_deformable_im2col(
    const T* data_im_, const T* data_offset_, const T* data_mask_,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kenerl_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, T* data_col_,
    cudaStream_t stream) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  modulated_deformable_im2col_gpu_kernel<T>
      <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
          num_kernels, data_im_, data_offset_, data_mask_, height_im, width_im,
          kernel_h, kenerl_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
          dilation_w, channel_per_deformable_group, batch_size, channels,
          deformable_group, height_col, width_col, data_col_);

  cudaCheckError();
}

template <typename scalar_t>
__global__ void output_add_bias_kernel(scalar_t* output, const scalar_t* bias,
                                       size_t step_batch, size_t step_channel,
                                       size_t n) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    output[index] += bias[(index % step_batch) / step_channel];
  }
}

template <typename scalar_t>
static void output_add_bias(scalar_t* output, const scalar_t* bias,
                            size_t batch, size_t channel, size_t height,
                            size_t width, cudaStream_t stream) {
  size_t step_channel = height * width;
  size_t step_batch = step_channel * channel;
  size_t n = step_batch * batch;
  output_add_bias_kernel<<<GET_BLOCKS(n), THREADS_PER_BLOCK, 0, stream>>>(
      output, bias, step_batch, step_channel, n);
}

template <typename scalar_t>
void ModulatedDeformConvForwardCUDAKernelLauncher(
    const scalar_t* input, const scalar_t* weight, const scalar_t* bias,
    const scalar_t* offset, const scalar_t* mask, scalar_t* output,
    void* workspace, int batch, int channels, int height, int width,
    int channels_out, int kernel_w, int kernel_h, int stride_w, int stride_h,
    int pad_w, int pad_h, int dilation_w, int dilation_h, int group,
    int deformable_group, int im2col_step, cublasHandle_t cublas_handle,
    cudaStream_t stream) {
  size_t sizeof_dtype = sizeof(scalar_t);
  bool with_bias = (bias != nullptr);

  im2col_step = std::min(int(batch), im2col_step);
  assert(batch % im2col_step == 0);
  const int channels_kernel = channels / group;

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  scalar_t* columns = (scalar_t*)workspace;

  const size_t input_step = channels * height * width;
  const size_t offset_step =
      deformable_group * kernel_h * kernel_w * 2 * height_out * width_out;
  const size_t mask_step =
      deformable_group * kernel_h * kernel_w * height_out * width_out;
  const size_t out_step = channels_out * height_out * width_out;
  const size_t out_group_step = out_step / group;
  const size_t col_g_step =
      channels * kernel_w * kernel_h / group * height_out * width_out;
  const size_t weight_g_step =
      channels_out / group * channels / group * kernel_h * kernel_w;

  const int m = channels_out / group;
  const int n = height_out * width_out;
  const int k = channels / group * kernel_h * kernel_w;
  scalar_t alpha = 1.;
  scalar_t beta = 0.;

  for (int b = 0; b < batch; b++) {
    const scalar_t* input_start = input + b * input_step;
    const scalar_t* offset_start = offset + b * offset_step;
    const scalar_t* mask_start = mask + b * mask_step;
    trt_modulated_deformable_im2col<scalar_t>(
        input_start, offset_start, mask_start, 1, channels, height, width,
        height_out, width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, dilation_h, dilation_w, deformable_group, columns, stream);

    for (int g = 0; g < group; g++) {
      const scalar_t* weight_start = weight + g * weight_g_step;
      scalar_t* col_start = columns + g * col_g_step;
      scalar_t* out_buffer_start = output + b * out_step + g * out_group_step;

      // cudaMemsetAsync(out_buffer_start, 0, 1, stream);
      cublasGemmWrap<scalar_t>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                               &alpha, col_start, n, weight_start, k, &beta,
                               out_buffer_start, n);
      cudaCheckError();
    }
  }

  if (with_bias) {
    output_add_bias<scalar_t>(output, bias, batch, channels_out, height_out,
                              width_out, stream);
  }
}

void ModulatedDeformConvForwardCUDAKernelLauncher_float(
    const float* input, const float* weight, const float* bias,
    const float* offset, const float* mask, float* output, void* workspace,
    int batch, int channels, int height, int width, int channels_out,
    int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w,
    int pad_h, int dilation_w, int dilation_h, int group, int deformable_group,
    int im2col_step, cublasHandle_t cublas_handle, cudaStream_t stream) {
  ModulatedDeformConvForwardCUDAKernelLauncher<float>(
      input, weight, bias, offset, mask, output, workspace, batch, channels,
      height, width, channels_out, kernel_w, kernel_h, stride_w, stride_h,
      pad_w, pad_h, dilation_w, dilation_h, group, deformable_group,
      im2col_step, cublas_handle, stream);
}

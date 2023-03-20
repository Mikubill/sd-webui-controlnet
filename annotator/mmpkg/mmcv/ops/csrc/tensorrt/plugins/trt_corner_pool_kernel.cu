// Copyright (c) OpenMMLab. All rights reserved
#include "common_cuda_helper.hpp"
#include "trt_cuda_helper.cuh"
#include "trt_plugin_helper.hpp"

template <typename scalar_t>
__global__ void top_bottom_pool_kernel(const scalar_t *input, scalar_t *output,
                                       const int batch_size, const int channels,
                                       const int height, const int width,
                                       const int pool_type) {
  const int nthreads = batch_size * channels * width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n_idx = index / (channels * width);  // batch
    int w_idx = index % width;               // width
    int c_idx = (index / width) % channels;  // channels
    int offset_n = n_idx * channels * width * height;
    int offset_n_c = offset_n + c_idx * width * height;
    int direction = -1;            // in [-1, 1], default for TopPool
    int index_start = height - 2;  // default for TopPool
    // pool_type in [0, 1]
    if (pool_type == 0) {
      // TopPool
      // directly copy the most bottom value from input to output
      output[offset_n_c + (height - 1) * width + w_idx] =
          input[offset_n_c + (height - 1) * width + w_idx];
    } else {
      // BottomPool
      // directly copy the most top value from input to output
      output[offset_n_c + w_idx] = input[offset_n_c + w_idx];
      index_start = 1;
      direction = 1;
    }
    // do pool
    for (int h = index_start; h >= 0 && h < height; h += direction) {
      output[offset_n_c + h * width + w_idx] =
          max(output[offset_n_c + (h - direction) * width + w_idx],
              input[offset_n_c + h * width + w_idx]);
    }
  }
}

template <typename scalar_t>
__global__ void left_right_pool_kernel(const scalar_t *input, scalar_t *output,
                                       const int batch_size, const int channels,
                                       const int height, const int width,
                                       const int pool_type) {
  const int nthreads = batch_size * channels * height;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n_idx = index / (channels * height);  // batch
    int h_idx = index % height;               // height
    int c_idx = (index / height) % channels;  // channels
    int offset_n = n_idx * channels * width * height;
    int offset_n_c = offset_n + c_idx * width * height;
    int offset_n_c_h = offset_n_c + h_idx * width;
    int direction = -1;           // in [-1, 1], default for LeftPool
    int index_start = width - 2;  // default for LeftPool
    // pool_type in [2, 3]
    if (pool_type == 2) {
      // LeftPool
      // directly copy the most right value from input to output
      output[offset_n_c_h + width - 1] = input[offset_n_c_h + width - 1];
    } else {
      // RightPool
      // directly copy the most left value from input to output
      output[offset_n_c_h] = input[offset_n_c_h];
      index_start = 1;
      direction = 1;
    }
    // do pool
    for (int w = index_start; w >= 0 && w < width; w += direction) {
      output[offset_n_c_h + w] =
          max(output[offset_n_c_h + w - direction], input[offset_n_c_h + w]);
    }
  }
}

template <typename scalar_t>
void CornerPoolForwardLauncher(const scalar_t *input, scalar_t *output,
                               const int batch_size, const int channels,
                               const int height, const int width,
                               const int pool_type, cudaStream_t stream) {
  int nthreads = -1, col_block = -1;

  switch (pool_type) {
    case 0:
    case 1:
      nthreads = batch_size * channels * width;
      col_block = GET_BLOCKS(nthreads, THREADS_PER_BLOCK);
      top_bottom_pool_kernel<scalar_t>
          <<<col_block, THREADS_PER_BLOCK, 0, stream>>>(
              input, output, batch_size, channels, height, width, pool_type);
      break;
    case 2:
    case 3:
      nthreads = batch_size * channels * height;
      col_block = GET_BLOCKS(nthreads, THREADS_PER_BLOCK);
      left_right_pool_kernel<scalar_t>
          <<<col_block, THREADS_PER_BLOCK, 0, stream>>>(
              input, output, batch_size, channels, height, width, pool_type);
      break;
  }
}

void CornerPoolForwardLauncher_float(const float *input, float *output,
                                     const int batch_size, const int channels,
                                     const int height, const int width,
                                     const int pool_type, cudaStream_t stream) {
  CornerPoolForwardLauncher<float>(input, output, batch_size, channels, height,
                                   width, pool_type, stream);
}

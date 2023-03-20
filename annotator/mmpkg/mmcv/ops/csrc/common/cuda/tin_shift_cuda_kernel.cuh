// Copyright (c) OpenMMLab. All rights reserved
#ifndef TIN_SHIFT_CUDA_KERNEL_CUH
#define TIN_SHIFT_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void tin_shift_forward_cuda_kernel(
    const int nthreads, const T* input, const int* shift, T* output,
    const int batch_size, const int channels, const int t_size,
    const int hw_size, const int group_size, const int group_channel) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int hw_index = index % hw_size;
    const int j = (index / hw_size) % channels;

    const int n_index = (index / hw_size / channels) % batch_size;
    int group_id = j / group_channel;
    int t_shift = shift[n_index * group_size + group_id];
    int offset = n_index * t_size * hw_size * channels + hw_size * j + hw_index;
    for (int i = 0; i < t_size; i++) {
      int now_t = i + t_shift;
      int data_id = i * hw_size * channels + offset;
      if (now_t < 0 || now_t >= t_size) {
        continue;
      }
      int out_id = now_t * hw_size * channels + offset;
      output[out_id] = input[data_id];
    }
  }
}

template <typename T>
__global__ void tin_shift_backward_cuda_kernel(
    const int nthreads, const T* input, const int* shift, T* output,
    const int batch_size, const int channels, const int t_size,
    const int hw_size, const int group_size, const int group_channel) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int hw_index = index % hw_size;
    const int j = (index / hw_size) % channels;

    const int n_index = (index / hw_size / channels) % batch_size;
    int group_id = j / group_channel;
    int t_shift = shift[n_index * group_size + group_id];
    int offset = n_index * t_size * hw_size * channels + hw_size * j + hw_index;
    for (int i = 0; i < t_size; i++) {
      int now_t = i + t_shift;
      int data_id = i * hw_size * channels + offset;
      if (now_t < 0 || now_t >= t_size) {
        continue;
      }
      int out_id = now_t * hw_size * channels + offset;
      output[out_id] = input[data_id];
    }
  }
}

#endif  // TIN_SHIFT_CUDA_KERNEL_CUH

// Copyright (c) OpenMMLab. All rights reserved
#ifndef SOFTMAX_FOCAL_LOSS_CUDA_KERNEL_CUH
#define SOFTMAX_FOCAL_LOSS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void softmax_focal_loss_forward_cuda_kernel(
    const int nthreads, const T* softmax, const int64_t* target,
    const T* weight, T* output, const T gamma, const T alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int64_t label = target[index];
    T pred = softmax[index * num_classes + label];

    if (label >= 0) {
      output[index] =
          -alpha * pow((T)1. - pred, gamma) * log(max(pred, (T)FLT_MIN));
    } else {
      output[index] = 0;
    }
    if (weight != NULL) {
      output[index] *= weight[label];
    }
  }
}

template <typename T>
__global__ void softmax_focal_loss_backward_cuda1_kernel(
    const int nthreads, const T* softmax, const int64_t* target,
    const T* weight, T* buff, const T gamma, const T alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int64_t label = target[index];
    T pred = softmax[index * num_classes + label];

    if (label >= 0) {
      buff[index] = alpha * (-pow((T)1. - pred, gamma) +
                             gamma * pow((T)1. - pred, gamma - 1) * pred *
                                 log(max(pred, (T)FLT_MIN)));
    } else {
      buff[index] = 0;
    }
    if (weight != NULL) {
      buff[index] *= weight[label];
    }
  }
}

template <typename T>
__global__ void softmax_focal_loss_backward_cuda2_kernel(
    const int nthreads, const T* softmax, const int64_t* target, const T* buff,
    T* grad_input, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;
    int64_t label = target[n];

    if (label >= 0) {
      T flag = (label == c ? (T)1. : (T)0.);
      grad_input[index] = buff[n] * (flag - softmax[index]);
    } else {
      grad_input[index] = 0;
    }
  }
}

#endif  // SOFTMAX_FOCAL_LOSS_CUDA_KERNEL_CUH

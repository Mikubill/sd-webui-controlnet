// Copyright (c) OpenMMLab. All rights reserved
#ifndef SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH
#define SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void sigmoid_focal_loss_forward_cuda_kernel(
    const int nthreads, const T* input, const int64_t* target, const T* weight,
    T* output, const T gamma, const T alpha, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    T flag_p = (t == c);
    T flag_n = (t != c);

    // p = sigmoid(x) = 1. / 1. + expf(-x)
    T p = (T)1. / ((T)1. + expf(-input[index]));

    // (1 - p)**gamma * log(p)
    T term_p = pow(((T)1. - p), gamma) * log(max(p, (T)FLT_MIN));
    // p**gamma * log(1 - p)
    T term_n = pow(p, gamma) * log(max((T)1. - p, (T)FLT_MIN));

    output[index] = (T)0.;
    output[index] += -flag_p * alpha * term_p;
    output[index] += -flag_n * ((T)1. - alpha) * term_n;
    if (weight != NULL) {
      output[index] *= weight[t];
    }
  }
}

template <typename T>
__global__ void sigmoid_focal_loss_backward_cuda_kernel(
    const int nthreads, const T* input, const int64_t* target, const T* weight,
    T* grad_input, const T gamma, const T alpha, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    T flag_p = (t == c);
    T flag_n = (t != c);

    // p = sigmoid(x) = 1. / 1. + expf(-x)
    T p = (T)1. / ((T)1. + exp(-input[index]));

    // (1 - p)**gamma * (1 - p - gamma*p*log(p))
    T term_p = pow(((T)1. - p), gamma) *
               ((T)1. - p - (gamma * p * log(max(p, (T)FLT_MIN))));
    // p**gamma * (gamma * (1 - p) * log(1 - p) - p)
    T term_n = pow(p, gamma) *
               (gamma * ((T)1. - p) * log(max((T)1. - p, (T)FLT_MIN)) - p);

    grad_input[index] = (T)0.;
    grad_input[index] += -flag_p * alpha * term_p;
    grad_input[index] += -flag_n * ((T)1. - alpha) * term_n;
    if (weight != NULL) {
      grad_input[index] *= weight[t];
    }
  }
}

#endif  // SIGMOID_FOCAL_LOSS_CUDA_KERNEL_CUH

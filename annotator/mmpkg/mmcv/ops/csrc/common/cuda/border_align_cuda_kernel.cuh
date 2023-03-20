// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/Megvii-BaseDetection/cvpods/blob/master/cvpods/layers/csrc/border_align/border_align_kernel.cu.
// the main difference: (1) use `argmax_idx` for fast computing of gradient
// during the backward. (2) `wh` is directly computed by `boxes`, rather than
// passing it as argument to forward or backward functions.

#ifndef BORDER_ALIGN_CUDA_KERNEL_CUH
#define BORDER_ALIGN_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT

enum BorderMode { Top = 0, Left = 1, Bottom = 2, Right = 3 };

/*** Forward ***/
template <typename T>
__global__ void border_align_forward_cuda_kernel(
    const int nthreads, const T* input, const T* boxes, T* output,
    int* argmax_idx, const int channels, const int box_size, const int height,
    const int width, const int pool_size) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (batch_idx, c_idx, box_idx) is an element paralleled for computing
    // output, and `extreme_idx` is in range [0,3]
    int batch_idx, c_idx, box_idx, extreme_idx, maxidx, *offset_argmax_idx;
    const T *offset_box, *offset_input, *offset_box_x;
    T *offset_output, box_width, box_height, stride, x_stride, y_stride, x, y,
        val, maxval;

    extreme_idx = threadIdx.y;
    // shape (N, C, box_size, 4) for output
    batch_idx = index / channels / box_size;
    // shape (N, box_size, 4) for boxes
    box_idx = index % box_size + batch_idx * box_size;
    c_idx = (index / box_size) % channels;

    offset_box = boxes + box_idx * 4;
    box_width = *(offset_box + 2) - *offset_box;
    box_height = *(offset_box + 3) - *(offset_box + 1);
    offset_output = output + index * 4 + extreme_idx;
    offset_argmax_idx = argmax_idx + index * 4 + extreme_idx;
    // shape (N, 4C, h, w) for input.
    // [0,C) for top feature, [C,2C) for left feature,
    // [2C,3C) for bottom feature, [3C,4C) for right feature
    offset_input =
        input + (batch_idx * channels * 4 + extreme_idx * channels + c_idx) *
                    height * width;

    // extreme_idx in [0,1] -> offset_box_x indexed at x1
    // extreme_idx in [2,3] -> offset_box_x indexed at x2
    offset_box_x = offset_box + extreme_idx / 2 * 2;

    // (x1,y1) or (x2,y2) for (x,y)
    x = *offset_box_x;
    y = *(offset_box_x + 1);

    switch (extreme_idx) {
      // top
      case BorderMode::Top:
        stride = box_width / pool_size;
        x_stride = stride;
        y_stride = 0;
        break;
      // left
      case BorderMode::Left:
        stride = box_height / pool_size;
        x_stride = 0;
        y_stride = stride;
        break;
      // bottom
      case BorderMode::Bottom:
        stride = box_width / pool_size;
        x_stride = -stride;
        y_stride = 0;
        break;
      // right
      case BorderMode::Right:
        stride = box_height / pool_size;
        x_stride = 0;
        y_stride = -stride;
        break;
    }

    // initialize maxval and maxidx with the start position (e.g. (x1,y1) or
    // (x2,y2))
    maxval = bilinear_interpolate(offset_input, height, width, y, x, index);
    maxidx = 0;

    // do max_pool along the border
    for (int i = 1; i <= pool_size; i++) {
      x += x_stride;
      y += y_stride;
      val = bilinear_interpolate(offset_input, height, width, y, x, index);
      if (val > maxval) {
        maxval = val;
        maxidx = i;
      }
    }

    // update output and argmax_idx
    *offset_output = maxval;
    *offset_argmax_idx = maxidx;
  }
}

/*** Backward ***/
template <typename T>
__global__ void border_align_backward_cuda_kernel(
    const int nthreads, const T* grad_output, const T* boxes,
    const int* argmax_idx, T* grad_input, const int channels,
    const int box_size, const int height, const int width,
    const int pool_size) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (batch_idx, c_idx, box_idx) is an element paralleled for computing
    // output, and `extreme_idx` is in range [0,3]
    int batch_idx, c_idx, box_idx, extreme_idx;
    const int* offset_argmax_idx;
    const T *offset_grad_output, *offset_box, *offset_box_x;
    T *offset_grad_input, box_width, box_height, stride, x_stride, y_stride, x,
        y;

    extreme_idx = threadIdx.y;
    batch_idx = index / channels / box_size;
    box_idx = index % box_size + batch_idx * box_size;
    c_idx = (index / box_size) % channels;

    offset_box = boxes + box_idx * 4;
    box_width = *(offset_box + 2) - *offset_box;
    box_height = *(offset_box + 3) - *(offset_box + 1);
    offset_grad_output = grad_output + index * 4 + extreme_idx;
    offset_argmax_idx = argmax_idx + index * 4 + extreme_idx;
    // [0,C) for top feature grad, [C,2C) for left feature grad,
    // [2C,3C) for bottom feature grad, [3C,4C) for right feature grad
    offset_grad_input = grad_input + (batch_idx * channels * 4 +
                                      extreme_idx * channels + c_idx) *
                                         height * width;

    // extreme_idx in [0,1] -> offset_box_x indexed at x1
    // extreme_idx in [2,3] -> offset_box_x indexed at x2
    offset_box_x = offset_box + extreme_idx / 2 * 2;

    switch (extreme_idx) {
      // top
      case BorderMode::Top:
        stride = box_width / pool_size;
        x_stride = stride;
        y_stride = 0;
        break;
      // left
      case BorderMode::Left:
        stride = box_height / pool_size;
        x_stride = 0;
        y_stride = stride;
        break;
      // bottom
      case BorderMode::Bottom:
        stride = box_width / pool_size;
        x_stride = -stride;
        y_stride = 0;
        break;
      // right
      case BorderMode::Right:
        stride = box_height / pool_size;
        x_stride = 0;
        y_stride = -stride;
        break;
    }

    // get position (x,y) which has maximum value during forward
    x = *offset_box_x;
    y = *(offset_box_x + 1);
    x += x_stride * (T)(*offset_argmax_idx);
    y += y_stride * (T)(*offset_argmax_idx);

    T w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;
    bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low,
                                  x_high, y_low, y_high, index);

    // update grad_output
    atomicAdd(offset_grad_input + y_low * width + x_low,
              *offset_grad_output * w1);
    atomicAdd(offset_grad_input + y_low * width + x_high,
              *offset_grad_output * w2);
    atomicAdd(offset_grad_input + y_high * width + x_low,
              *offset_grad_output * w3);
    atomicAdd(offset_grad_input + y_high * width + x_high,
              *offset_grad_output * w4);
  }
}

#endif  // BORDER_ALIGN_CUDA_KERNEL_CUH

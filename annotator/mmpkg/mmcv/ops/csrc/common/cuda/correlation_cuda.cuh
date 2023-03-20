// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/ClementPinard/Pytorch-Correlation-extension/blob/master/Correlation_Module/correlation_cuda_kernel.cu
// Original licence: Under MIT License

#ifndef CORRELATION_CUDA
#define CORRELATION_CUDA

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#include <cuda.h>
#include <cuda_runtime.h>
// Using <torch/extension.h> is recommended in the official documentation in
// https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-the-c-op.
// However, we use <torch/types.h> for compatibility with CUDA 9.0
// Read https://github.com/pytorch/extension-cpp/issues/35 for more details.
#include <torch/types.h>

#include <iostream>
#include <vector>

using namespace torch;

#define TensorAcc4R PackedTensorAccessor32<scalar_t, 4, RestrictPtrTraits>
#define TensorAcc5R PackedTensorAccessor32<scalar_t, 5, RestrictPtrTraits>
#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < H && y >= 0 && y < W)

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void correlation_forward_cuda_kernel(
    const TensorAcc4R rInput1, const TensorAcc4R rInput2, TensorAcc5R output,
    int kH, int kW, int patchH, int patchW, int padH, int padW, int dilationH,
    int dilationW, int dilation_patchH, int dilation_patchW, int dH, int dW,
    int oH, int oW) {
  const int iH = rInput1.size(1);
  const int iW = rInput1.size(2);
  const int C = rInput1.size(3);

  const int n = blockIdx.x;
  const int h = blockIdx.y * blockDim.y + threadIdx.y;
  const int w = blockIdx.z * blockDim.z + threadIdx.z;

  if (h >= oH || w >= oW) return;

  const int thread = threadIdx.x;

  const int start_i = -padH + h * dH;
  const int start_j = -padW + w * dW;

  const int patchRadH = dilation_patchH * (patchH - 1) / 2;
  const int patchRadW = dilation_patchW * (patchW - 1) / 2;

  for (int ph = 0; ph < patchH; ++ph) {
    int ph_dilated = ph * dilation_patchH - patchRadH;
    for (int pw = 0; pw < patchW; ++pw) {
      int pw_dilated = pw * dilation_patchW - patchRadW;
      scalar_t prod_sum = 0.0f;
      for (int i = 0; i < kH; ++i) {
        int i1 = start_i + i * dilationH;
        int i2 = i1 + ph_dilated;
        if (WITHIN_BOUNDS(i1, i2, iH, iH)) {
          for (int j = 0; j < kW; ++j) {
            int j1 = start_j + j * dilationW;
            int j2 = j1 + pw_dilated;
            if (WITHIN_BOUNDS(j1, j2, iW, iW)) {
              for (int c = thread; c < C; c += WARP_SIZE) {
                scalar_t v1 = rInput1[n][i1][j1][c];
                scalar_t v2 = rInput2[n][i2][j2][c];
                prod_sum += v1 * v2;
              }
            }
          }
        }
      }
      // accumulate
      for (int offset = 16; offset > 0; offset /= 2)
#ifdef MMCV_WITH_HIP
        prod_sum += __shfl_down(float(prod_sum), offset);
#else
        prod_sum += __shfl_down_sync(FULL_MASK, float(prod_sum), offset);
#endif
      if (thread == 0) {
        output[n][ph][pw][h][w] = prod_sum;
      }
    }
  }
}

template <typename scalar_t>
__global__ void correlation_backward_cuda_kernel_input1(
    const TensorAcc5R grad_output, const TensorAcc4R input2,
    TensorAcc4R grad_input1, const int kH, const int kW, const int patchH,
    const int patchW, const int padH, const int padW, const int dilationH,
    const int dilationW, const int dilation_patchH, const int dilation_patchW,
    const int dH, const int dW) {
  const int iH = input2.size(1);
  const int iW = input2.size(2);
  const int C = input2.size(3);

  const int H = grad_output.size(3);
  const int W = grad_output.size(4);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;

  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;

  const int h_2 = h + padH;
  const int w_2 = w + padW;
  const int min_h = h_2 - kH * dilationH;
  const int min_w = w_2 - kW * dilationW;

  extern __shared__ __align__(sizeof(4)) unsigned char grad_cache_char[];
  scalar_t *grad_cache = reinterpret_cast<scalar_t *>(grad_cache_char);
  for (int i = threadIdx.x; i < patchH * patchW; i += blockDim.x) {
    const int ph = i / patchW;
    const int pw = i % patchW;
    int i1 = h + dilation_patchH * (ph - patchRadH);
    int j1 = w + dilation_patchW * (pw - patchRadW);

    if (WITHIN_BOUNDS(i1, j1, iH, iW)) {
      scalar_t grad_val = 0.0f;
      for (int h_3 = h_2; h_3 > min_h; h_3 -= dilationH) {
        int i2 = (h_3) / dH;
        if (i2 * dH != h_3) continue;
        for (int w_3 = w_2; w_3 > min_w; w_3 -= dilationW) {
          int j2 = (w_3) / dW;
          if (j2 * dW != w_3) continue;
          if (WITHIN_BOUNDS(i2, j2, H, W)) {
            grad_val += grad_output[n][ph][pw][i2][j2];
          }
        }
      }
      grad_cache[i] = grad_val;
    }
  }
  __syncthreads();

  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    scalar_t grad_input_val = 0.0f;
    for (int ph = 0; ph < patchH; ++ph) {
      int i1 = h + dilation_patchH * (ph - patchRadH);
      for (int pw = 0; pw < patchW; ++pw) {
        int j1 = w + dilation_patchW * (pw - patchRadW);
        if (WITHIN_BOUNDS(i1, j1, iH, iW)) {
          grad_input_val += input2[n][i1][j1][c] * grad_cache[ph * patchW + pw];
        }
      }
    }
    grad_input1[n][c][h][w] = grad_input_val;
  }
}

template <typename scalar_t>
__global__ void correlation_backward_cuda_kernel_input2(
    const TensorAcc5R grad_output, const TensorAcc4R input1,
    TensorAcc4R grad_input2, int kH, int kW, int patchH, int patchW, int padH,
    int padW, int dilationH, int dilationW, int dilation_patchH,
    int dilation_patchW, int dH, int dW) {
  const int iH = input1.size(1);
  const int iW = input1.size(2);
  const int C = input1.size(3);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;

  const int H = grad_output.size(3);
  const int W = grad_output.size(4);

  const int dilatedKH = kH * dilationH;
  const int dilatedKW = kW * dilationW;

  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;

  extern __shared__ __align__(sizeof(4)) unsigned char grad_cache_char[];
  scalar_t *grad_cache = reinterpret_cast<scalar_t *>(grad_cache_char);
  for (int i = threadIdx.x; i < patchH * patchW; i += blockDim.x) {
    const int ph = i / patchW;
    const int pw = i % patchW;
    int i1 = h - dilation_patchH * (ph - patchRadH);
    int j1 = w - dilation_patchW * (pw - patchRadW);

    if (WITHIN_BOUNDS(i1, j1, iH, iW)) {
      scalar_t grad_val = 0.0f;

      const int h_2 = i1 + padH;
      const int w_2 = j1 + padW;
      const int min_h = h_2 - dilatedKH;
      const int min_w = w_2 - dilatedKW;

      for (int h_3 = h_2; h_3 > min_h; h_3 -= dilationH) {
        int i2 = (h_3) / dH;
        if (i2 * dH != h_3) continue;
        for (int w_3 = w_2; w_3 > min_w; w_3 -= dilationW) {
          int j2 = (w_3) / dW;
          if (j2 * dW != w_3) continue;
          if (WITHIN_BOUNDS(i2, j2, H, W)) {
            grad_val += grad_output[n][ph][pw][i2][j2];
          }
        }
      }
      grad_cache[i] = grad_val;
    }
  }
  __syncthreads();

  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    scalar_t grad_input_val = 0.0f;
    for (int ph = 0; ph < patchH; ++ph) {
      int i1 = h - dilation_patchH * (ph - patchRadH);
      for (int pw = 0; pw < patchW; ++pw) {
        int j1 = w - dilation_patchW * (pw - patchRadW);
        if (WITHIN_BOUNDS(i1, j1, iH, iW)) {
          grad_input_val += input1[n][i1][j1][c] * grad_cache[ph * patchW + pw];
        }
      }
    }
    grad_input2[n][c][h][w] = grad_input_val;
  }
}
#endif

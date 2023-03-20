/*
 * Copyright (c) 2019, SenseTime.
 */

#ifndef INCLUDE_PARROTS_DARRAY_CUDAWARPFUNCTION_CUH_
#define INCLUDE_PARROTS_DARRAY_CUDAWARPFUNCTION_CUH_

#ifndef __CUDACC__
#error cudawarpfunction.cuh should only be included by .cu files
#endif
#include <cuda.h>

#include <parrots/foundation/common.hpp>

#ifdef PARROTS_USE_HALF
#include <cuda_fp16.h>
#endif
#ifdef __CUDA_ARCH__
#define CUDA_INTRINSIC_FUNC(Expr) Expr
#else
#define CUDA_INTRINSIC_FUNC(Expr)
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

#ifdef PARROTS_USE_HALF

#if CUDA_VERSION < 9000

__device__ inline float16 __shfl(float16 var, int srcLane, int width) {
  CUDA_INTRINSIC_FUNC(return __shfl(var.y, srcLane, width););
}

__device__ inline float16 __shfl_up(float16 var, unsigned delta, int width) {
  CUDA_INTRINSIC_FUNC(return __shfl_up(var.y, delta, width););
}

__device__ inline float16 __shfl_down(float16 var, unsigned delta, int width) {
  CUDA_INTRINSIC_FUNC(return __shfl_down(var.y, delta, width););
}

__device__ inline float16 __shfl_xor(float16 var, int laneMask, int width) {
  CUDA_INTRINSIC_FUNC(return __shfl_xor(var.y, laneMask, width););
}

#else  // CUDA_VERSION >= 9000

__device__ inline float16 __shfl_sync(unsigned mask, float16 var, int srcLane,
                                      int width = warpSize) {
  CUDA_INTRINSIC_FUNC(float16 r; r.y = __shfl_sync(mask, var.y, srcLane, width);
                      return r;);
}

__device__ inline float16 __shfl_up_sync(unsigned mask, float16 var,
                                         unsigned delta, int width = warpSize) {
  CUDA_INTRINSIC_FUNC(
      float16 r; r.y = __shfl_up_sync(mask, var.y, delta, width); return r;);
}

__device__ inline float16 __shfl_down_sync(unsigned mask, float16 var,
                                           unsigned delta,
                                           int width = warpSize) {
  CUDA_INTRINSIC_FUNC(
      float16 r; r.y = __shfl_down_sync(mask, var.y, delta, width); return r;);
}

__device__ inline float16 __shfl_xor_sync(unsigned mask, float16 var,
                                          int laneMask, int width) {
  CUDA_INTRINSIC_FUNC(float16 r;
                      r.y = __shfl_xor_sync(mask, var.y, laneMask, width);
                      return r;);
}

#endif  // CUDA_VERSION < 9000

#endif  // PARROTS_USE_HALF

// warp shuffle interface with a dummy mask
#if CUDA_VERSION < 9000

template <typename T>
__device__ inline T __shfl_sync(unsigned mask, T var, int srcLane,
                                int width = warpSize) {
  CUDA_INTRINSIC_FUNC(return __shfl(var, srcLane, width););
}

template <typename T>
__device__ inline T __shfl_up_sync(unsigned mask, T var, unsigned delta,
                                   int width = warpSize) {
  CUDA_INTRINSIC_FUNC(return __shfl_up(var, delta, width););
}

template <typename T>
__device__ inline T __shfl_down_sync(unsigned mask, T var, unsigned delta,
                                     int width = warpSize) {
  CUDA_INTRINSIC_FUNC(return __shfl_down(var, delta, width););
}

template <typename T>
__device__ inline T __shfl_xor_sync(unsigned mask, T var, int laneMask,
                                    int width = warpSize) {
  CUDA_INTRINSIC_FUNC(return __shfl_xor(var, laneMask, width););
}

#endif  // CUDA_VERSION < 9000

#endif  // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

#endif  // INCLUDE_PARROTS_DARRAY_CUDAWARPFUNCTION_CUH_

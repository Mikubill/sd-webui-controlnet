// Copyright (c) OpenMMLab. All rights reserved
#ifndef SCATTER_POINTS_CUDA_KERNEL_CUH
#define SCATTER_POINTS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;
int const maxGridDim = 50000;

__device__ __forceinline__ static void reduceMax(float *address, float val) {
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old || __int_as_float(old) < val);
}

__device__ __forceinline__ static void reduceMax(double *address, double val) {
  unsigned long long *address_as_ull =
      reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old || __longlong_as_double(old) < val);
}

// get rid of meaningless warnings when compiling host code
#ifdef MMCV_WITH_HIP
__device__ __forceinline__ static void reduceAdd(float *address, float val) {
  atomicAdd(address, val);
}
__device__ __forceinline__ static void reduceAdd(double *address, double val) {
  atomicAdd(address, val);
}
#else
#ifdef __CUDA_ARCH__
__device__ __forceinline__ static void reduceAdd(float *address, float val) {
#if (__CUDA_ARCH__ < 200)
#ifdef _MSC_VER
#pragma message( \
    "compute capability lower than 2.x. fall back to use CAS version of atomicAdd for float32")
#else
#warning \
    "compute capability lower than 2.x. fall back to use CAS version of atomicAdd for float32"
#endif
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(val + __int_as_float(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}

__device__ __forceinline__ static void reduceAdd(double *address, double val) {
#if (__CUDA_ARCH__ < 600)
#ifdef _MSC_VER
#pragma message( \
    "compute capability lower than 6.x. fall back to use CAS version of atomicAdd for float64")
#else
#warning \
    "compute capability lower than 6.x. fall back to use CAS version of atomicAdd for float64"
#endif
  unsigned long long *address_as_ull =
      reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}
#endif  // __CUDA_ARCH__
#endif  // MMCV_WITH_HIP

template <typename T>
__global__ void feats_reduce_kernel(
    const T *feats, const int32_t *coors_map,
    T *reduced_feats,  // shall be 0 at initialization
    const int num_input, const int num_feats, const reduce_t reduce_type) {
  CUDA_1D_KERNEL_LOOP(x, num_input) {
    int32_t reduce_to = coors_map[x];
    if (reduce_to == -1) continue;

    const T *feats_offset = feats + x * num_feats;
    T *reduced_feats_offset = reduced_feats + reduce_to * num_feats;
    if (reduce_type == reduce_t::MAX) {
      for (int i = 0; i < num_feats; i++) {
        reduceMax(&reduced_feats_offset[i], feats_offset[i]);
      }
    } else {
      for (int i = 0; i < num_feats; i++) {
        reduceAdd(&reduced_feats_offset[i], feats_offset[i]);
      }
    }
  }
}

template <typename T>
__global__ void add_reduce_traceback_grad_kernel(
    T *grad_feats, const T *grad_reduced_feats, const int32_t *coors_map,
    const int32_t *reduce_count, const int num_input, const int num_feats,
    const reduce_t reduce_type) {
  CUDA_1D_KERNEL_LOOP(x, num_input) {
    int32_t reduce_to = coors_map[x];
    if (reduce_to == -1) {
      continue;
    }

    const int input_offset = x * num_feats;
    T *grad_feats_offset = grad_feats + input_offset;
    const int reduced_offset = reduce_to * num_feats;
    const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    if (reduce_type == reduce_t::SUM) {
      for (int i = 0; i < num_feats; i++) {
        grad_feats_offset[i] = grad_reduced_feats_offset[i];
      }
    } else if (reduce_type == reduce_t::MEAN) {
      for (int i = 0; i < num_feats; i++) {
        grad_feats_offset[i] = grad_reduced_feats_offset[i] /
                               static_cast<T>(reduce_count[reduce_to]);
      }
    }
  }
}

template <typename T>
__global__ void max_reduce_traceback_scatter_idx_kernel(
    const T *feats, const T *reduced_feats, int32_t *reduce_from,
    const int32_t *coors_map, const int num_input, const int num_feats) {
  CUDA_1D_KERNEL_LOOP(x, num_input) {
    int32_t reduce_to = coors_map[x];

    const int input_offset = x * num_feats;
    const T *feats_offset = feats + input_offset;

    if (reduce_to == -1) {
      continue;
    }

    const int reduced_offset = reduce_to * num_feats;
    const T *reduced_feats_offset = reduced_feats + reduced_offset;
    int32_t *reduce_from_offset = reduce_from + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      if (feats_offset[i] == reduced_feats_offset[i]) {
        atomicMin(&reduce_from_offset[i], static_cast<int32_t>(x));
      }
    }
  }
}

template <typename T>
__global__ void max_reduce_scatter_grad_kernel(T *grad_feats,
                                               const T *grad_reduced_feats,
                                               const int32_t *reduce_from,
                                               const int num_reduced,
                                               const int num_feats) {
  CUDA_1D_KERNEL_LOOP(x, num_reduced) {
    const int reduced_offset = x * num_feats;
    const int32_t *scatter_to_offset = reduce_from + reduced_offset;
    const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      grad_feats[scatter_to_offset[i] * num_feats + i] =
          grad_reduced_feats_offset[i];
    }
  }
}

#endif  // SCATTER_POINTS_CUDA_KERNEL_CUH

// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/csuhan/s2anet/blob/master/mmdet/ops/orn/src/cuda/ActiveRotatingFilter_cuda.cu
#ifndef ACTIVE_ROTATED_FILTER_CUDA_KERNEL_CUH
#define ACTIVE_ROTATED_FILTER_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename scalar_t>
__global__ void active_rotated_filter_forward_cuda_kernel(
    const int nthreads, const scalar_t* weight_data, const int* indices_data,
    const int num_input_planes, const int num_output_planes,
    const int num_orientations, const int num_rotations, const int nEntry,
    scalar_t* output_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int l = index % nEntry;
    int j = (index / nEntry) % num_input_planes;
    int i = index / nEntry / num_input_planes;
    int k;
    scalar_t val = *(weight_data + index);
    for (k = 0; k < num_rotations; k++) {
      int idx = (int)(*(indices_data + l * num_rotations + k)) - 1;
      scalar_t* target = output_data +
                         i * (num_rotations * num_input_planes * nEntry) +
                         k * (num_input_planes * nEntry) + j * (nEntry) + idx;
      *target = val;
    }
  }
}

template <typename scalar_t>
__global__ void active_rotated_filter_backward_cuda_kernel(
    const int nthreads, const scalar_t* gradWeight_data,
    const int* indices_data, const int num_input_planes,
    const int num_output_planes, const int num_orientations,
    const int num_rotations, const int nEntry, scalar_t* weight_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int l = index % nEntry;
    int j = (index / nEntry) % num_input_planes;
    int i = index / nEntry / num_input_planes;
    int k;
    scalar_t* val = weight_data + index;
    *val = 0;
    scalar_t tmp = 0;
    for (k = 0; k < num_rotations; k++) {
      int idx = (int)(*(indices_data + l * num_rotations + k)) - 1;
      scalar_t target =
          *(gradWeight_data + i * (num_rotations * num_input_planes * nEntry) +
            k * (num_input_planes * nEntry) + j * (nEntry) + idx);
      tmp = tmp + target;
    }
    *val = tmp;
  }
}
#endif  // ACTIVE_ROTATED_FILTER_CUDA_KERNEL_CUH

// Copyright (c) OpenMMLab. All rights reserved
#ifndef FURTHEST_POINT_SAMPLE_CUDA_KERNEL_CUH
#define FURTHEST_POINT_SAMPLE_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_forward_cuda_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
  // dataset: (B, N, 3)
  // tmp: (B, N)
  // output:
  //      idx: (B, M)

  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      // if (mag <= 1e-3)
      // continue;

      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

#pragma unroll
    for (int block_size_thres = 1024; block_size_thres >= 2;
         block_size_thres >>= 1) {
      const int tid_thres = block_size_thres / 2;
      if (block_size >= block_size_thres && tid < tid_thres) {
        __update(dists, dists_i, tid, tid + tid_thres);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}

// Modified from
// https://github.com/qiqihaer/3DSSD-pytorch/blob/master/lib/pointnet2/src/sampling_gpu.cu
template <unsigned int block_size>
__global__ void furthest_point_sampling_with_dist_forward_cuda_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
  // dataset: (B, N, N)
  // tmp: (B, N)
  // output:
  //      idx: (B, M)

  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * n;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    // float x1 = dataset[old * 3 + 0];
    // float y1 = dataset[old * 3 + 1];
    // float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      // float x2, y2, z2;
      // x2 = dataset[k * 3 + 0];
      // y2 = dataset[k * 3 + 1];
      // z2 = dataset[k * 3 + 2];

      // float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) *
      // (z2 - z1);
      float d = dataset[old * n + k];

      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

#pragma unroll
    for (int block_size_thres = 1024; block_size_thres >= 2;
         block_size_thres >>= 1) {
      const int tid_thres = block_size_thres / 2;
      if (block_size >= block_size_thres && tid < tid_thres) {
        __update(dists, dists_i, tid, tid + tid_thres);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}

#endif  // FURTHEST_POINT_SAMPLE_CUDA_KERNEL_CUH

// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu
#ifndef STACK_BALL_QUERY_CUDA_KERNEL_CUH
#define STACK_BALL_QUERY_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void stack_ball_query_forward_cuda_kernel(
    int B, int M, float radius, int nsample, const T *new_xyz,
    const int *new_xyz_batch_cnt, const T *xyz, const int *xyz_batch_cnt,
    int *idx) {
  // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
  // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
  // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
  // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
  // output:
  //      idx: (M, nsample)
  const T *cur_xyz = xyz;
  int *cur_idx = idx;
  CUDA_1D_KERNEL_LOOP(pt_idx, M) {
    int bs_idx = 0;
    for (int pt_cnt = 0; bs_idx < B; bs_idx++) {
      pt_cnt += new_xyz_batch_cnt[bs_idx];
      if (pt_idx < pt_cnt) break;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

    const T *new_xyz_p = new_xyz + pt_idx * 3;
    cur_xyz += xyz_batch_start_idx * 3;
    cur_idx += pt_idx * nsample;

    float radius2 = radius * radius;
    T new_x = new_xyz_p[0];
    T new_y = new_xyz_p[1];
    T new_z = new_xyz_p[2];
    int n = xyz_batch_cnt[bs_idx];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
      T x = cur_xyz[k * 3 + 0];
      T y = cur_xyz[k * 3 + 1];
      T z = cur_xyz[k * 3 + 2];
      T d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
             (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            cur_idx[l] = k;
          }
        }
        cur_idx[cnt] = k;
        ++cnt;
        if (cnt >= nsample) break;
      }
    }
    if (cnt == 0) cur_idx[0] = -1;
  }
}

#endif  // STACK_BALL_QUERY_CUDA_KERNEL_CUH

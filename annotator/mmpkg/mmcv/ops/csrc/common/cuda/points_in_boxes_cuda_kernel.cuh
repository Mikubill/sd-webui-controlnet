// Copyright (c) OpenMMLab. All rights reserved
#ifndef POINT_IN_BOXES_CUDA_KERNEL_CUH
#define POINT_IN_BOXES_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__device__ inline void lidar_to_local_coords(T shift_x, T shift_y, T rz,
                                             T &local_x, T &local_y) {
  T cosa = cos(-rz), sina = sin(-rz);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

template <typename T>
__device__ inline int check_pt_in_box3d(const T *pt, const T *box3d, T &local_x,
                                        T &local_y) {
  // param pt: (x, y, z)
  // param box3d: (cx, cy, cz, x_size, y_size, z_size, rz) in LiDAR coordinate,
  // cz in the bottom center
  T x = pt[0], y = pt[1], z = pt[2];
  T cx = box3d[0], cy = box3d[1], cz = box3d[2];
  T x_size = box3d[3], y_size = box3d[4], z_size = box3d[5], rz = box3d[6];
  cz += z_size /
        2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > z_size / 2.0) return 0;
  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
  float in_flag = (local_x > -x_size / 2.0) & (local_x < x_size / 2.0) &
                  (local_y > -y_size / 2.0) & (local_y < y_size / 2.0);
  return in_flag;
}

template <typename T>
__global__ void points_in_boxes_part_forward_cuda_kernel(
    int batch_size, int boxes_num, int pts_num, const T *boxes, const T *pts,
    int *box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center, each box DO NOT overlaps params pts:
  // (B, npoints, 3) [x, y, z] in LiDAR coordinate params boxes_idx_of_points:
  // (B, npoints), default -1

  int bs_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, pts_num) {
    if (bs_idx >= batch_size) return;

    boxes += bs_idx * boxes_num * 7;
    pts += bs_idx * pts_num * 3 + pt_idx * 3;
    box_idx_of_points += bs_idx * pts_num + pt_idx;

    T local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    for (int k = 0; k < boxes_num; k++) {
      cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
      if (cur_in_flag) {
        box_idx_of_points[0] = k;
        break;
      }
    }
  }
}

template <typename T>
__global__ void points_in_boxes_all_forward_cuda_kernel(
    int batch_size, int boxes_num, int pts_num, const T *boxes, const T *pts,
    int *box_idx_of_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center, each box DO NOT overlaps params pts:
  // (B, npoints, 3) [x, y, z] in LiDAR coordinate params boxes_idx_of_points:
  // (B, npoints), default -1

  int bs_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, pts_num) {
    if (bs_idx >= batch_size) return;

    boxes += bs_idx * boxes_num * 7;
    pts += bs_idx * pts_num * 3 + pt_idx * 3;
    box_idx_of_points += bs_idx * pts_num * boxes_num + pt_idx * boxes_num;

    T local_x = 0, local_y = 0;
    for (int k = 0; k < boxes_num; k++) {
      const int cur_in_flag =
          check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
      if (cur_in_flag) {
        box_idx_of_points[k] = 1;
      }
    }
  }
}

#endif  // POINT_IN_BOXES_CUDA_KERNEL_CUH

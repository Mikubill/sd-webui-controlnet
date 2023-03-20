// Copyright (c) OpenMMLab. All rights reserved
#ifndef ROIAWARE_POOL3D_CUDA_KERNEL_CUH
#define ROIAWARE_POOL3D_CUDA_KERNEL_CUH

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
__global__ void generate_pts_mask_for_box3d(int boxes_num, int pts_num,
                                            int out_x, int out_y, int out_z,
                                            const T *rois, const T *pts,
                                            int *pts_mask) {
  // params rois: (N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate params pts: (npoints, 3) [x, y, z] params pts_mask: (N,
  // npoints): -1 means point does not in this box, otherwise: encode (x_idxs,
  // y_idxs, z_idxs) by binary bit
  int box_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(pt_idx, pts_num) {
    if (box_idx >= boxes_num) return;

    pts += pt_idx * 3;
    rois += box_idx * 7;
    pts_mask += box_idx * pts_num + pt_idx;

    T local_x = 0, local_y = 0;
    int cur_in_flag = check_pt_in_box3d(pts, rois, local_x, local_y);

    pts_mask[0] = -1;
    if (cur_in_flag > 0) {
      T local_z = pts[2] - rois[2];
      T x_size = rois[3], y_size = rois[4], z_size = rois[5];

      T x_res = x_size / out_x;
      T y_res = y_size / out_y;
      T z_res = z_size / out_z;

      unsigned int x_idx = int((local_x + x_size / 2) / x_res);
      unsigned int y_idx = int((local_y + y_size / 2) / y_res);
      unsigned int z_idx = int(local_z / z_res);

      x_idx = min(max(x_idx, 0), out_x - 1);
      y_idx = min(max(y_idx, 0), out_y - 1);
      z_idx = min(max(z_idx, 0), out_z - 1);

      unsigned int idx_encoding = (x_idx << 16) + (y_idx << 8) + z_idx;

      pts_mask[0] = idx_encoding;
    }
  }
}

template <typename T>
__global__ void collect_inside_pts_for_box3d(int boxes_num, int pts_num,
                                             int max_pts_each_voxel, int out_x,
                                             int out_y, int out_z,
                                             const int *pts_mask,
                                             T *pts_idx_of_voxels) {
  // params pts_mask: (N, npoints)  0 or 1
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  CUDA_1D_KERNEL_LOOP(box_idx, boxes_num) {
    int max_num_pts = max_pts_each_voxel - 1;  // index 0 is the counter
    pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel;

    for (int k = 0; k < pts_num; k++) {
      if (pts_mask[box_idx * pts_num + k] != -1) {
        unsigned int idx_encoding = pts_mask[box_idx * pts_num + k];
        unsigned int x_idx = (idx_encoding >> 16) & 0xFF;
        unsigned int y_idx = (idx_encoding >> 8) & 0xFF;
        unsigned int z_idx = idx_encoding & 0xFF;
        unsigned int base_offset = x_idx * out_y * out_z * max_pts_each_voxel +
                                   y_idx * out_z * max_pts_each_voxel +
                                   z_idx * max_pts_each_voxel;
        unsigned int cnt = pts_idx_of_voxels[base_offset];
        if (cnt < max_num_pts) {
          pts_idx_of_voxels[base_offset + cnt + 1] = k;
          pts_idx_of_voxels[base_offset]++;
        }
      }
    }
  }
}

template <typename T>
__global__ void roiaware_maxpool3d(int boxes_num, int pts_num, int channels,
                                   int max_pts_each_voxel, int out_x, int out_y,
                                   int out_z, const T *pts_feature,
                                   const int *pts_idx_of_voxels,
                                   T *pooled_features, int *argmax) {
  // params pts_feature: (npoints, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel),
  // index 0 is the counter params pooled_features: (N, out_x, out_y, out_z, C)
  // params argmax: (N, out_x, out_y, out_z, C)

  int box_idx = blockIdx.z;
  int channel_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(voxel_idx_flat, out_x * out_y * out_z) {
    int x_idx = voxel_idx_flat / (out_y * out_z);
    int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
    int z_idx = voxel_idx_flat % out_z;
    if (box_idx >= boxes_num || channel_idx >= channels) return;

    int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
    pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel +
                         offset_base * max_pts_each_voxel;
    pooled_features += box_idx * out_x * out_y * out_z * channels +
                       offset_base * channels + channel_idx;
    argmax += box_idx * out_x * out_y * out_z * channels +
              offset_base * channels + channel_idx;

    int argmax_idx = -1;
    float max_val = -1e50;

    int total_pts = pts_idx_of_voxels[0];

    for (int k = 1; k <= total_pts; k++) {
      if (pts_feature[pts_idx_of_voxels[k] * channels + channel_idx] >
          max_val) {
        max_val = pts_feature[pts_idx_of_voxels[k] * channels + channel_idx];
        argmax_idx = pts_idx_of_voxels[k];
      }
    }

    if (argmax_idx != -1) {
      pooled_features[0] = max_val;
    }
    argmax[0] = argmax_idx;
  }
}

template <typename T>
__global__ void roiaware_avgpool3d(int boxes_num, int pts_num, int channels,
                                   int max_pts_each_voxel, int out_x, int out_y,
                                   int out_z, const T *pts_feature,
                                   const int *pts_idx_of_voxels,
                                   T *pooled_features) {
  // params pts_feature: (npoints, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel),
  // index 0 is the counter params pooled_features: (N, out_x, out_y, out_z, C)
  // params argmax: (N, out_x, out_y, out_z, C)

  int box_idx = blockIdx.z;
  int channel_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(voxel_idx_flat, out_x * out_y * out_z) {
    int x_idx = voxel_idx_flat / (out_y * out_z);
    int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
    int z_idx = voxel_idx_flat % out_z;
    if (box_idx >= boxes_num || channel_idx >= channels) return;

    int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
    pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel +
                         offset_base * max_pts_each_voxel;
    pooled_features += box_idx * out_x * out_y * out_z * channels +
                       offset_base * channels + channel_idx;

    float sum_val = 0;
    int total_pts = pts_idx_of_voxels[0];

    for (int k = 1; k <= total_pts; k++) {
      sum_val += pts_feature[pts_idx_of_voxels[k] * channels + channel_idx];
    }

    if (total_pts > 0) {
      pooled_features[0] = sum_val / total_pts;
    }
  }
}

template <typename T>
__global__ void roiaware_maxpool3d_backward(int boxes_num, int channels,
                                            int out_x, int out_y, int out_z,
                                            const int *argmax,
                                            const T *grad_out, T *grad_in) {
  // params argmax: (N, out_x, out_y, out_z, C)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value

  int box_idx = blockIdx.z;
  int channel_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(voxel_idx_flat, out_x * out_y * out_z) {
    int x_idx = voxel_idx_flat / (out_y * out_z);
    int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
    int z_idx = voxel_idx_flat % out_z;
    if (box_idx >= boxes_num || channel_idx >= channels) return;

    int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
    argmax += box_idx * out_x * out_y * out_z * channels +
              offset_base * channels + channel_idx;
    grad_out += box_idx * out_x * out_y * out_z * channels +
                offset_base * channels + channel_idx;

    if (argmax[0] == -1) return;

    atomicAdd(grad_in + argmax[0] * channels + channel_idx, grad_out[0] * 1);
  }
}

template <typename T>
__global__ void roiaware_avgpool3d_backward(int boxes_num, int channels,
                                            int out_x, int out_y, int out_z,
                                            int max_pts_each_voxel,
                                            const int *pts_idx_of_voxels,
                                            const T *grad_out, T *grad_in) {
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value

  int box_idx = blockIdx.z;
  int channel_idx = blockIdx.y;
  CUDA_1D_KERNEL_LOOP(voxel_idx_flat, out_x * out_y * out_z) {
    int x_idx = voxel_idx_flat / (out_y * out_z);
    int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
    int z_idx = voxel_idx_flat % out_z;
    if (box_idx >= boxes_num || channel_idx >= channels) return;

    int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
    pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel +
                         offset_base * max_pts_each_voxel;
    grad_out += box_idx * out_x * out_y * out_z * channels +
                offset_base * channels + channel_idx;

    int total_pts = pts_idx_of_voxels[0];
    float cur_grad = 1 / fmaxf(float(total_pts), 1.0);
    for (int k = 1; k <= total_pts; k++) {
      atomicAdd(grad_in + pts_idx_of_voxels[k] * channels + channel_idx,
                grad_out[0] * cur_grad);
    }
  }
}

#endif  // ROIAWARE_POOL3D_CUDA_KERNEL_CUH

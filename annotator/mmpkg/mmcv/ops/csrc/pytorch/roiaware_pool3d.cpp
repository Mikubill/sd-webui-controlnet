#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void roiaware_pool3d_forward_impl(int boxes_num, int pts_num, int channels,
                                  int max_pts_each_voxel, int out_x, int out_y,
                                  int out_z, const Tensor rois,
                                  const Tensor pts, const Tensor pts_feature,
                                  Tensor argmax, Tensor pts_idx_of_voxels,
                                  Tensor pooled_features, int pool_method) {
  DISPATCH_DEVICE_IMPL(roiaware_pool3d_forward_impl, boxes_num, pts_num,
                       channels, max_pts_each_voxel, out_x, out_y, out_z, rois,
                       pts, pts_feature, argmax, pts_idx_of_voxels,
                       pooled_features, pool_method);
}

void roiaware_pool3d_backward_impl(int boxes_num, int out_x, int out_y,
                                   int out_z, int channels,
                                   int max_pts_each_voxel,
                                   const Tensor pts_idx_of_voxels,
                                   const Tensor argmax, const Tensor grad_out,
                                   Tensor grad_in, int pool_method) {
  DISPATCH_DEVICE_IMPL(roiaware_pool3d_backward_impl, boxes_num, out_x, out_y,
                       out_z, channels, max_pts_each_voxel, pts_idx_of_voxels,
                       argmax, grad_out, grad_in, pool_method);
}

void roiaware_pool3d_forward(Tensor rois, Tensor pts, Tensor pts_feature,
                             Tensor argmax, Tensor pts_idx_of_voxels,
                             Tensor pooled_features, int pool_method) {
  // params rois: (N, 7) [x, y, z, x_size, y_size, z_size, ry] in LiDAR
  // coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate
  // params pts_feature: (npoints, C)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params pooled_features: (N, out_x, out_y, out_z, C)
  // params pool_method: 0: max_pool 1: avg_pool
  int boxes_num = rois.size(0);
  int pts_num = pts.size(0);
  int channels = pts_feature.size(1);
  int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
  int out_x = pts_idx_of_voxels.size(1);
  int out_y = pts_idx_of_voxels.size(2);
  int out_z = pts_idx_of_voxels.size(3);
  assert((out_x < 256) && (out_y < 256) &&
         (out_z < 256));  // we encode index with 8bit

  roiaware_pool3d_forward_impl(boxes_num, pts_num, channels, max_pts_each_voxel,
                               out_x, out_y, out_z, rois, pts, pts_feature,
                               argmax, pts_idx_of_voxels, pooled_features,
                               pool_method);
}

void roiaware_pool3d_backward(Tensor pts_idx_of_voxels, Tensor argmax,
                              Tensor grad_out, Tensor grad_in,
                              int pool_method) {
  // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
  // params argmax: (N, out_x, out_y, out_z, C)
  // params grad_out: (N, out_x, out_y, out_z, C)
  // params grad_in: (npoints, C), return value
  // params pool_method: 0: max_pool 1: avg_pool
  int boxes_num = pts_idx_of_voxels.size(0);
  int out_x = pts_idx_of_voxels.size(1);
  int out_y = pts_idx_of_voxels.size(2);
  int out_z = pts_idx_of_voxels.size(3);
  int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
  int channels = grad_out.size(4);

  roiaware_pool3d_backward_impl(boxes_num, out_x, out_y, out_z, channels,
                                max_pts_each_voxel, pts_idx_of_voxels, argmax,
                                grad_out, grad_in, pool_method);
}

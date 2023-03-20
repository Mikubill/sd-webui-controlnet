/*
Modified from
https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp
Point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void roipoint_pool3d_forward_impl(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const Tensor xyz, const Tensor boxes3d,
                                  const Tensor pts_feature,
                                  Tensor pooled_features,
                                  Tensor pooled_empty_flag) {
  DISPATCH_DEVICE_IMPL(roipoint_pool3d_forward_impl, batch_size, pts_num,
                       boxes_num, feature_in_len, sampled_pts_num, xyz, boxes3d,
                       pts_feature, pooled_features, pooled_empty_flag);
}

void roipoint_pool3d_forward(Tensor xyz, Tensor boxes3d, Tensor pts_feature,
                             Tensor pooled_features, Tensor pooled_empty_flag) {
  // params xyz: (B, N, 3)
  // params boxes3d: (B, M, 7)
  // params pts_feature: (B, N, C)
  // params pooled_features: (B, M, 512, 3+C)
  // params pooled_empty_flag: (B, M)
  int batch_size = xyz.size(0);
  int pts_num = xyz.size(1);
  int boxes_num = boxes3d.size(1);
  int feature_in_len = pts_feature.size(2);
  int sampled_pts_num = pooled_features.size(2);

  roipoint_pool3d_forward_impl(batch_size, pts_num, boxes_num, feature_in_len,
                               sampled_pts_num, xyz, boxes3d, pts_feature,
                               pooled_features, pooled_empty_flag);
}

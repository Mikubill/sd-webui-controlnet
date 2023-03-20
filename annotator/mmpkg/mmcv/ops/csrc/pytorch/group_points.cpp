// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void group_points_forward_impl(int b, int c, int n, int npoints, int nsample,
                               const Tensor points, const Tensor idx,
                               Tensor out) {
  DISPATCH_DEVICE_IMPL(group_points_forward_impl, b, c, n, npoints, nsample,
                       points, idx, out);
}

void group_points_backward_impl(int b, int c, int n, int npoints, int nsample,
                                const Tensor grad_out, const Tensor idx,
                                Tensor grad_points) {
  DISPATCH_DEVICE_IMPL(group_points_backward_impl, b, c, n, npoints, nsample,
                       grad_out, idx, grad_points);
}

void group_points_forward(Tensor points_tensor, Tensor idx_tensor,
                          Tensor out_tensor, int b, int c, int n, int npoints,
                          int nsample) {
  DISPATCH_DEVICE_IMPL(group_points_forward_impl, b, c, n, npoints, nsample,
                       points_tensor, idx_tensor, out_tensor);
}

void group_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                           Tensor grad_points_tensor, int b, int c, int n,
                           int npoints, int nsample) {
  group_points_backward_impl(b, c, n, npoints, nsample, grad_out_tensor,
                             idx_tensor, grad_points_tensor);
}

void stack_group_points_backward_impl(int b, int c, int m, int n, int nsample,
                                      const Tensor grad_out_tensor,
                                      const Tensor idx_tensor,
                                      const Tensor idx_batch_cnt_tensor,
                                      const Tensor features_batch_cnt_tensor,
                                      Tensor grad_features_tensor) {
  DISPATCH_DEVICE_IMPL(stack_group_points_backward_impl, b, c, m, n, nsample,
                       grad_out_tensor, idx_tensor, idx_batch_cnt_tensor,
                       features_batch_cnt_tensor, grad_features_tensor);
}

void stack_group_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                                 Tensor idx_batch_cnt_tensor,
                                 Tensor features_batch_cnt_tensor,
                                 Tensor grad_features_tensor, int b, int c,
                                 int m, int n, int nsample) {
  stack_group_points_backward_impl(
      b, c, m, n, nsample, grad_out_tensor, idx_tensor, idx_batch_cnt_tensor,
      features_batch_cnt_tensor, grad_features_tensor);
}

void stack_group_points_forward_impl(int b, int c, int m, int nsample,
                                     const Tensor features_tensor,
                                     const Tensor features_batch_cnt_tensor,
                                     const Tensor idx_tensor,
                                     const Tensor idx_batch_cnt_tensor,
                                     Tensor out_tensor) {
  DISPATCH_DEVICE_IMPL(stack_group_points_forward_impl, b, c, m, nsample,
                       features_tensor, features_batch_cnt_tensor, idx_tensor,
                       idx_batch_cnt_tensor, out_tensor);
}

void stack_group_points_forward(Tensor features_tensor,
                                Tensor features_batch_cnt_tensor,
                                Tensor idx_tensor, Tensor idx_batch_cnt_tensor,
                                Tensor out_tensor, int b, int c, int m,
                                int nsample) {
  DISPATCH_DEVICE_IMPL(stack_group_points_forward_impl, b, c, m, nsample,
                       features_tensor, features_batch_cnt_tensor, idx_tensor,
                       idx_batch_cnt_tensor, out_tensor);
}

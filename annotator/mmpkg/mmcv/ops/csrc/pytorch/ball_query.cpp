// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void ball_query_forward_impl(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx) {
  DISPATCH_DEVICE_IMPL(ball_query_forward_impl, b, n, m, min_radius, max_radius,
                       nsample, new_xyz, xyz, idx);
}

void ball_query_forward(Tensor new_xyz_tensor, Tensor xyz_tensor,
                        Tensor idx_tensor, int b, int n, int m,
                        float min_radius, float max_radius, int nsample) {
  ball_query_forward_impl(b, n, m, min_radius, max_radius, nsample,
                          new_xyz_tensor, xyz_tensor, idx_tensor);
}

void stack_ball_query_forward_impl(float max_radius, int nsample,
                                   const Tensor new_xyz,
                                   const Tensor new_xyz_batch_cnt,
                                   const Tensor xyz, const Tensor xyz_batch_cnt,
                                   Tensor idx) {
  DISPATCH_DEVICE_IMPL(stack_ball_query_forward_impl, max_radius, nsample,
                       new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
}

void stack_ball_query_forward(Tensor new_xyz_tensor, Tensor new_xyz_batch_cnt,
                              Tensor xyz_tensor, Tensor xyz_batch_cnt,
                              Tensor idx_tensor, float max_radius,
                              int nsample) {
  stack_ball_query_forward_impl(max_radius, nsample, new_xyz_tensor,
                                new_xyz_batch_cnt, xyz_tensor, xyz_batch_cnt,
                                idx_tensor);
}

// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void furthest_point_sampling_forward_impl(Tensor points_tensor,
                                          Tensor temp_tensor, Tensor idx_tensor,
                                          int b, int n, int m) {
  DISPATCH_DEVICE_IMPL(furthest_point_sampling_forward_impl, points_tensor,
                       temp_tensor, idx_tensor, b, n, m);
}

void furthest_point_sampling_with_dist_forward_impl(Tensor points_tensor,
                                                    Tensor temp_tensor,
                                                    Tensor idx_tensor, int b,
                                                    int n, int m) {
  DISPATCH_DEVICE_IMPL(furthest_point_sampling_with_dist_forward_impl,
                       points_tensor, temp_tensor, idx_tensor, b, n, m);
}

void furthest_point_sampling_forward(Tensor points_tensor, Tensor temp_tensor,
                                     Tensor idx_tensor, int b, int n, int m) {
  furthest_point_sampling_forward_impl(points_tensor, temp_tensor, idx_tensor,
                                       b, n, m);
}

void furthest_point_sampling_with_dist_forward(Tensor points_tensor,
                                               Tensor temp_tensor,
                                               Tensor idx_tensor, int b, int n,
                                               int m) {
  furthest_point_sampling_with_dist_forward_impl(points_tensor, temp_tensor,
                                                 idx_tensor, b, n, m);
}

// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void three_interpolate_forward_impl(int b, int c, int m, int n,
                                    const Tensor points, const Tensor idx,
                                    const Tensor weight, Tensor out) {
  DISPATCH_DEVICE_IMPL(three_interpolate_forward_impl, b, c, m, n, points, idx,
                       weight, out);
}

void three_interpolate_backward_impl(int b, int c, int n, int m,
                                     const Tensor grad_out, const Tensor idx,
                                     const Tensor weight, Tensor grad_points) {
  DISPATCH_DEVICE_IMPL(three_interpolate_backward_impl, b, c, n, m, grad_out,
                       idx, weight, grad_points);
}

void three_interpolate_forward(Tensor points_tensor, Tensor idx_tensor,
                               Tensor weight_tensor, Tensor out_tensor, int b,
                               int c, int m, int n) {
  three_interpolate_forward_impl(b, c, m, n, points_tensor, idx_tensor,
                                 weight_tensor, out_tensor);
}

void three_interpolate_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                                Tensor weight_tensor, Tensor grad_points_tensor,
                                int b, int c, int n, int m) {
  three_interpolate_backward_impl(b, c, n, m, grad_out_tensor, idx_tensor,
                                  weight_tensor, grad_points_tensor);
}

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void gather_points_forward_impl(int b, int c, int n, int npoints,
                                const Tensor points, const Tensor idx,
                                Tensor out) {
  DISPATCH_DEVICE_IMPL(gather_points_forward_impl, b, c, n, npoints, points,
                       idx, out);
}

void gather_points_backward_impl(int b, int c, int n, int npoints,
                                 const Tensor grad_out, const Tensor idx,
                                 Tensor grad_points) {
  DISPATCH_DEVICE_IMPL(gather_points_backward_impl, b, c, n, npoints, grad_out,
                       idx, grad_points);
}

void gather_points_forward(Tensor points_tensor, Tensor idx_tensor,
                           Tensor out_tensor, int b, int c, int n,
                           int npoints) {
  gather_points_forward_impl(b, c, n, npoints, points_tensor, idx_tensor,
                             out_tensor);
}

void gather_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                            Tensor grad_points_tensor, int b, int c, int n,
                            int npoints) {
  gather_points_backward_impl(b, c, n, npoints, grad_out_tensor, idx_tensor,
                              grad_points_tensor);
}

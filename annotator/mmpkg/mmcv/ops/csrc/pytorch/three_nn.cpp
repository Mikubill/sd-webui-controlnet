// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void three_nn_forward_impl(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx) {
  DISPATCH_DEVICE_IMPL(three_nn_forward_impl, b, n, m, unknown, known, dist2,
                       idx);
}

void three_nn_forward(Tensor unknown_tensor, Tensor known_tensor,
                      Tensor dist2_tensor, Tensor idx_tensor, int b, int n,
                      int m) {
  three_nn_forward_impl(b, n, m, unknown_tensor, known_tensor, dist2_tensor,
                        idx_tensor);
}

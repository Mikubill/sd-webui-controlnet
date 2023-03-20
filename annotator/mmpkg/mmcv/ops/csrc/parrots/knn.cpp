// Modified from
// https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void knn_forward_impl(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2) {
  DISPATCH_DEVICE_IMPL(knn_forward_impl, b, n, m, nsample, xyz, new_xyz, idx,
                       dist2);
}

void knn_forward(Tensor xyz_tensor, Tensor new_xyz_tensor, Tensor idx_tensor,
                 Tensor dist2_tensor, int b, int n, int m, int nsample) {
  knn_forward_impl(b, n, m, nsample, xyz_tensor, new_xyz_tensor, idx_tensor,
                   dist2_tensor);
}

// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/chrdiller/pyTorchChamferDistance/blob/master/chamfer_distance/chamfer_distance.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void chamfer_distance_forward_impl(const Tensor xyz1, const Tensor xyz2,
                                   const Tensor dist1, const Tensor dist2,
                                   const Tensor idx1, const Tensor idx2) {
  DISPATCH_DEVICE_IMPL(chamfer_distance_forward_impl, xyz1, xyz2, dist1, dist2,
                       idx1, idx2);
}

void chamfer_distance_backward_impl(const Tensor xyz1, const Tensor xyz2,
                                    Tensor idx1, Tensor idx2, Tensor graddist1,
                                    Tensor graddist2, Tensor gradxyz1,
                                    Tensor gradxyz2) {
  DISPATCH_DEVICE_IMPL(chamfer_distance_backward_impl, xyz1, xyz2, idx1, idx2,
                       graddist1, graddist2, gradxyz1, gradxyz2);
}

void chamfer_distance_forward(const Tensor xyz1, const Tensor xyz2,
                              const Tensor dist1, const Tensor dist2,
                              const Tensor idx1, const Tensor idx2) {
  chamfer_distance_forward_impl(xyz1, xyz2, dist1, dist2, idx1, idx2);
}

void chamfer_distance_backward(const Tensor xyz1, const Tensor xyz2,
                               Tensor idx1, Tensor idx2, Tensor graddist1,
                               Tensor graddist2, Tensor gradxyz1,
                               Tensor gradxyz2) {
  chamfer_distance_backward_impl(xyz1, xyz2, idx1, idx2, graddist1, graddist2,
                                 gradxyz1, gradxyz2);
}

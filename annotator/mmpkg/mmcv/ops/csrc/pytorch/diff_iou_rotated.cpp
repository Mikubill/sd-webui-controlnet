// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor diff_iou_rotated_sort_vertices_forward_impl(Tensor vertices, Tensor mask,
                                                   Tensor num_valid) {
  return DISPATCH_DEVICE_IMPL(diff_iou_rotated_sort_vertices_forward_impl,
                              vertices, mask, num_valid);
}

Tensor diff_iou_rotated_sort_vertices_forward(Tensor vertices, Tensor mask,
                                              Tensor num_valid) {
  return diff_iou_rotated_sort_vertices_forward_impl(vertices, mask, num_valid);
}

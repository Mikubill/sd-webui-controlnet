// Copyright (c) OpenMMLab. All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_impl(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const reduce_t reduce_type) {
  return DISPATCH_DEVICE_IMPL(dynamic_point_to_voxel_forward_impl, feats, coors,
                              reduce_type);
}

void dynamic_point_to_voxel_backward_impl(
    torch::Tensor &grad_feats, const torch::Tensor &grad_reduced_feats,
    const torch::Tensor &feats, const torch::Tensor &reduced_feats,
    const torch::Tensor &coors_idx, const torch::Tensor &reduce_count,
    const reduce_t reduce_type) {
  DISPATCH_DEVICE_IMPL(dynamic_point_to_voxel_backward_impl, grad_feats,
                       grad_reduced_feats, feats, reduced_feats, coors_idx,
                       reduce_count, reduce_type);
}

inline reduce_t convert_reduce_type(const std::string &reduce_type) {
  if (reduce_type == "max")
    return reduce_t::MAX;
  else if (reduce_type == "sum")
    return reduce_t::SUM;
  else if (reduce_type == "mean")
    return reduce_t::MEAN;
  else
    TORCH_CHECK(false, "do not support reduce type " + reduce_type)
  return reduce_t::SUM;
}

std::vector<torch::Tensor> dynamic_point_to_voxel_forward(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const std::string &reduce_type) {
  return dynamic_point_to_voxel_forward_impl(feats, coors,
                                             convert_reduce_type(reduce_type));
}

void dynamic_point_to_voxel_backward(torch::Tensor &grad_feats,
                                     const torch::Tensor &grad_reduced_feats,
                                     const torch::Tensor &feats,
                                     const torch::Tensor &reduced_feats,
                                     const torch::Tensor &coors_idx,
                                     const torch::Tensor &reduce_count,
                                     const std::string &reduce_type) {
  dynamic_point_to_voxel_backward_impl(grad_feats, grad_reduced_feats, feats,
                                       reduced_feats, coors_idx, reduce_count,
                                       convert_reduce_type(reduce_type));
}

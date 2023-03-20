// Copyright (c) OpenMMLab. All rights reserved
// It is modified from https://github.com/WenmuZhou/PAN.pytorch

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

std::vector<std::vector<float>> pixel_group_impl(
    Tensor score, Tensor mask, Tensor embedding, Tensor kernel_label,
    Tensor kernel_contour, int kernel_region_num, float dis_threshold) {
  return DISPATCH_DEVICE_IMPL(pixel_group_impl, score, mask, embedding,
                              kernel_label, kernel_contour, kernel_region_num,
                              dis_threshold);
}

std::vector<std::vector<float>> pixel_group(
    Tensor score, Tensor mask, Tensor embedding, Tensor kernel_label,
    Tensor kernel_contour, int kernel_region_num, float distance_threshold) {
  score = score.contiguous();
  mask = mask.contiguous();
  embedding = embedding.contiguous();
  kernel_label = kernel_label.contiguous();
  kernel_contour = kernel_contour.contiguous();

  return pixel_group_impl(score, mask, embedding, kernel_label, kernel_contour,
                          kernel_region_num, distance_threshold);
}

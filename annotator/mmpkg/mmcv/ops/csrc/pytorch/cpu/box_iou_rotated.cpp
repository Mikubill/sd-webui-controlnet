// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cpu.cpp
#include "box_iou_rotated_utils.hpp"
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

template <typename T>
void box_iou_rotated_cpu_kernel(const Tensor boxes1, const Tensor boxes2,
                                Tensor ious, const int mode_flag,
                                const bool aligned) {
  int output_size = ious.numel();
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  if (aligned) {
    for (int i = 0; i < output_size; i++) {
      ious[i] = single_box_iou_rotated<T>(boxes1[i].data_ptr<T>(),
                                          boxes2[i].data_ptr<T>(), mode_flag);
    }
  } else {
    for (int i = 0; i < num_boxes1; i++) {
      for (int j = 0; j < num_boxes2; j++) {
        ious[i * num_boxes2 + j] = single_box_iou_rotated<T>(
            boxes1[i].data_ptr<T>(), boxes2[j].data_ptr<T>(), mode_flag);
      }
    }
  }
}

void box_iou_rotated_cpu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned) {
  box_iou_rotated_cpu_kernel<float>(boxes1, boxes2, ious, mode_flag, aligned);
}

void box_iou_rotated_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);
REGISTER_DEVICE_IMPL(box_iou_rotated_impl, CPU, box_iou_rotated_cpu);

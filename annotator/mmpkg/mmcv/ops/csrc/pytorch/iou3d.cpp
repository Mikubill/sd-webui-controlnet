// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

void iou3d_boxes_overlap_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap) {
  DISPATCH_DEVICE_IMPL(iou3d_boxes_overlap_bev_forward_impl, num_a, boxes_a,
                       num_b, boxes_b, ans_overlap);
}

void iou3d_nms3d_forward_impl(const Tensor boxes, Tensor &keep,
                              Tensor &keep_num, float nms_overlap_thresh) {
  DISPATCH_DEVICE_IMPL(iou3d_nms3d_forward_impl, boxes, keep, keep_num,
                       nms_overlap_thresh);
}

void iou3d_nms3d_normal_forward_impl(const Tensor boxes, Tensor &keep,
                                     Tensor &keep_num,
                                     float nms_overlap_thresh) {
  DISPATCH_DEVICE_IMPL(iou3d_nms3d_normal_forward_impl, boxes, keep, keep_num,
                       nms_overlap_thresh);
}

void iou3d_boxes_overlap_bev_forward(Tensor boxes_a, Tensor boxes_b,
                                     Tensor ans_overlap) {
  // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params boxes_b: (M, 5)
  // params ans_overlap: (N, M)
  int num_a = boxes_a.size(0);
  int num_b = boxes_b.size(0);

  iou3d_boxes_overlap_bev_forward_impl(num_a, boxes_a, num_b, boxes_b,
                                       ans_overlap);
}

void iou3d_nms3d_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                         float nms_overlap_thresh) {
  // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params keep: (N)
  CHECK_CONTIGUOUS(boxes);
  CHECK_CONTIGUOUS(keep);

  iou3d_nms3d_forward_impl(boxes, keep, keep_num, nms_overlap_thresh);
}

void iou3d_nms3d_normal_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                                float nms_overlap_thresh) {
  // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params keep: (N)

  CHECK_CONTIGUOUS(boxes);
  CHECK_CONTIGUOUS(keep);

  iou3d_nms3d_normal_forward_impl(boxes, keep, keep_num, nms_overlap_thresh);
}

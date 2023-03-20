// Copyright (c) OpenMMLab. All rights reserved
#include "nms.h"

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>  // std::iota
#include <vector>

#include "../ort_mmcv_utils.h"

NmsKernel::NmsKernel(OrtApi api, const OrtKernelInfo *info)
    : api_(api), ort_(api_), info_(info) {
  iou_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "iou_threshold");
  offset_ = ort_.KernelInfoGetAttribute<int64_t>(info, "offset");

  // create allocator
  allocator_ = Ort::AllocatorWithDefaultOptions();
}

void NmsKernel::Compute(OrtKernelContext *context) {
  const float iou_threshold = iou_threshold_;
  const int64_t offset = offset_;

  const OrtValue *boxes = ort_.KernelContext_GetInput(context, 0);
  const float *boxes_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(boxes));
  const OrtValue *scores = ort_.KernelContext_GetInput(context, 1);
  const float *scores_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(scores));

  OrtTensorDimensions boxes_dim(ort_, boxes);
  OrtTensorDimensions scores_dim(ort_, scores);

  int64_t nboxes = boxes_dim[0];
  assert(boxes_dim[1] == 4);

  // allocate tmp memory
  float *tmp_boxes = (float *)allocator_.Alloc(sizeof(float) * nboxes * 4);
  float *sc = (float *)allocator_.Alloc(sizeof(float) * nboxes);
  float *areas = (float *)allocator_.Alloc(sizeof(float) * nboxes);
  bool *select = (bool *)allocator_.Alloc(sizeof(bool) * nboxes);
  for (int64_t i = 0; i < nboxes; i++) {
    select[i] = true;
  }

  memcpy(tmp_boxes, boxes_data, sizeof(float) * nboxes * 4);
  memcpy(sc, scores_data, sizeof(float) * nboxes);

  // sort scores
  std::vector<float> tmp_sc;
  for (int i = 0; i < nboxes; i++) {
    tmp_sc.push_back(sc[i]);
  }
  std::vector<int64_t> order(tmp_sc.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&tmp_sc](int64_t id1, int64_t id2) {
    return tmp_sc[id1] > tmp_sc[id2];
  });

  // area = (x2 - x1 + offset) * (y2 - y1 + offset)
  for (int64_t i = 0; i < nboxes; i++) {
    areas[i] = (tmp_boxes[i * 4 + 2] - tmp_boxes[i * 4 + 0] + offset) *
               (tmp_boxes[i * 4 + 3] - tmp_boxes[i * 4 + 1] + offset);
  }

  for (int64_t _i = 0; _i < nboxes; _i++) {
    if (select[_i] == false) continue;
    auto i = order[_i];
    auto ix1 = tmp_boxes[i * 4 + 0];
    auto iy1 = tmp_boxes[i * 4 + 1];
    auto ix2 = tmp_boxes[i * 4 + 2];
    auto iy2 = tmp_boxes[i * 4 + 3];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < nboxes; _j++) {
      if (select[_j] == false) continue;
      auto j = order[_j];
      auto xx1 = std::max(ix1, tmp_boxes[j * 4 + 0]);
      auto yy1 = std::max(iy1, tmp_boxes[j * 4 + 1]);
      auto xx2 = std::min(ix2, tmp_boxes[j * 4 + 2]);
      auto yy2 = std::min(iy2, tmp_boxes[j * 4 + 3]);

      auto w = std::max(0.f, xx2 - xx1 + offset);
      auto h = std::max(0.f, yy2 - yy1 + offset);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold) select[_j] = false;
    }
  }
  std::vector<int64_t> res_order;
  for (int i = 0; i < nboxes; i++) {
    if (select[i]) {
      res_order.push_back(order[i]);
    }
  }

  std::vector<int64_t> inds_dims({res_order.size()});

  OrtValue *res = ort_.KernelContext_GetOutput(context, 0, inds_dims.data(),
                                               inds_dims.size());
  int64_t *res_data = ort_.GetTensorMutableData<int64_t>(res);

  memcpy(res_data, res_order.data(), sizeof(int64_t) * res_order.size());
}

// Copyright (c) OpenMMLab. All rights reserved
#include "soft_nms.h"

#include <assert.h>

#include <algorithm>
#include <cmath>

#include "../ort_mmcv_utils.h"

SoftNmsKernel::SoftNmsKernel(OrtApi api, const OrtKernelInfo *info)
    : api_(api), ort_(api_), info_(info) {
  iou_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "iou_threshold");
  sigma_ = ort_.KernelInfoGetAttribute<float>(info, "sigma");
  min_score_ = ort_.KernelInfoGetAttribute<float>(info, "min_score");
  method_ = ort_.KernelInfoGetAttribute<int64_t>(info, "method");
  offset_ = ort_.KernelInfoGetAttribute<int64_t>(info, "offset");

  // create allocator
  allocator_ = Ort::AllocatorWithDefaultOptions();
}

void SoftNmsKernel::Compute(OrtKernelContext *context) {
  typedef float T;

  const T iou_threshold = T(iou_threshold_);
  const T sigma = T(sigma_);
  const T min_score = T(min_score_);
  const int method = int(method_);
  const T offset = T(offset_);

  const OrtValue *boxes = ort_.KernelContext_GetInput(context, 0);
  const T *boxes_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<T>(boxes));
  const OrtValue *scores = ort_.KernelContext_GetInput(context, 1);
  const T *scores_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<T>(scores));

  OrtTensorDimensions boxes_dim(ort_, boxes);
  OrtTensorDimensions scores_dim(ort_, scores);

  int64_t nboxes = boxes_dim[0];
  assert(boxes_dim[1] == 4);

  // allocate tmp memory
  T *tmp_boxes = (T *)allocator_.Alloc(sizeof(T) * nboxes * 4);
  T *x1 = tmp_boxes;
  T *y1 = tmp_boxes + 1;
  T *x2 = tmp_boxes + 2;
  T *y2 = tmp_boxes + 3;
  T *sc = (T *)allocator_.Alloc(sizeof(T) * nboxes);
  T *areas = (T *)allocator_.Alloc(sizeof(T) * nboxes);
  T *de = (T *)allocator_.Alloc(sizeof(T) * nboxes * 5);
  int64_t *inds = (int64_t *)allocator_.Alloc(sizeof(int64_t) * nboxes);

  memcpy(tmp_boxes, boxes_data, sizeof(T) * nboxes * 4);
  memcpy(sc, scores_data, sizeof(T) * nboxes);

  // init inds as arange(nboxes)
  std::generate(inds, inds + nboxes, [n = 0]() mutable { return n++; });

  // area = (x2-x1+offset)*(y2-y1+offset)
  for (int64_t i = 0; i < nboxes; i++) {
    areas[i] =
        (x2[i * 4] - x1[i * 4] + offset) * (y2[i * 4] - y1[i * 4] + offset);
  }

  int64_t pos = 0;

  for (int64_t i = 0; i < nboxes; i++) {
    auto max_score = sc[i];
    auto max_pos = i;

    pos = i + 1;
    // get max box
    while (pos < nboxes) {
      if (max_score < sc[pos]) {
        max_score = sc[pos];
        max_pos = pos;
      }
      pos = pos + 1;
    }
    // swap
    auto ix1 = de[i * 5 + 0] = x1[max_pos * 4];
    auto iy1 = de[i * 5 + 1] = y1[max_pos * 4];
    auto ix2 = de[i * 5 + 2] = x2[max_pos * 4];
    auto iy2 = de[i * 5 + 3] = y2[max_pos * 4];
    auto iscore = de[i * 5 + 4] = sc[max_pos];
    auto iarea = areas[max_pos];
    auto iind = inds[max_pos];
    x1[max_pos * 4] = x1[i * 4];
    y1[max_pos * 4] = y1[i * 4];
    x2[max_pos * 4] = x2[i * 4];
    y2[max_pos * 4] = y2[i * 4];
    sc[max_pos] = sc[i];
    areas[max_pos] = areas[i];
    inds[max_pos] = inds[i];
    x1[i * 4] = ix1;
    y1[i * 4] = iy1;
    x2[i * 4] = ix2;
    y2[i * 4] = iy2;
    sc[i] = iscore;
    areas[i] = iarea;
    inds[i] = iind;

    pos = i + 1;
    while (pos < nboxes) {
      auto xx1 = std::max(ix1, x1[pos * 4]);
      auto yy1 = std::max(iy1, y1[pos * 4]);
      auto xx2 = std::min(ix2, x2[pos * 4]);
      auto yy2 = std::min(iy2, y2[pos * 4]);

      auto w = std::max(0.f, xx2 - xx1 + offset);
      auto h = std::max(0.f, yy2 - yy1 + offset);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[pos] - inter);

      float weight = 1.;
      if (method == 0) {
        if (ovr >= iou_threshold) weight = 0;
      } else if (method == 1) {
        if (ovr >= iou_threshold) weight = 1 - ovr;
      } else if (method == 2) {
        weight = std::exp(-(ovr * ovr) / sigma);
      }
      sc[pos] *= weight;
      // if box score falls below threshold, discard the box by
      // swapping with last box update N
      if (sc[pos] < min_score) {
        x1[pos * 4] = x1[(nboxes - 1) * 4];
        y1[pos * 4] = y1[(nboxes - 1) * 4];
        x2[pos * 4] = x2[(nboxes - 1) * 4];
        y2[pos * 4] = y2[(nboxes - 1) * 4];
        sc[pos] = sc[nboxes - 1];
        areas[pos] = areas[nboxes - 1];
        inds[pos] = inds[nboxes - 1];
        nboxes = nboxes - 1;
        pos = pos - 1;
      }
      pos = pos + 1;
    }
  }

  std::vector<int64_t> dets_dim({nboxes, 5});
  OrtValue *dets = ort_.KernelContext_GetOutput(context, 0, dets_dim.data(),
                                                dets_dim.size());
  T *dets_data = ort_.GetTensorMutableData<T>(dets);

  std::vector<int64_t> inds_dim({nboxes});
  OrtValue *inds_ov = ort_.KernelContext_GetOutput(context, 1, inds_dim.data(),
                                                   inds_dim.size());
  int64_t *inds_data = ort_.GetTensorMutableData<int64_t>(inds_ov);

  memcpy(dets_data, de, sizeof(T) * nboxes * 5);
  memcpy(inds_data, inds, sizeof(int64_t) * nboxes);
}

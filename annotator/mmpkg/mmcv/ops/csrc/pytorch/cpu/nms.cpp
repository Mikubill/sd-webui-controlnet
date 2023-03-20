// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor nms_cpu(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }
  auto x1_t = boxes.select(1, 0).contiguous();
  auto y1_t = boxes.select(1, 1).contiguous();
  auto x2_t = boxes.select(1, 2).contiguous();
  auto y2_t = boxes.select(1, 3).contiguous();

  Tensor areas_t = (x2_t - x1_t + offset) * (y2_t - y1_t + offset);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto nboxes = boxes.size(0);
  Tensor select_t = at::ones({nboxes}, boxes.options().dtype(at::kBool));

  auto select = select_t.data_ptr<bool>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();

  for (int64_t _i = 0; _i < nboxes; _i++) {
    if (select[_i] == false) continue;
    auto i = order[_i];
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < nboxes; _j++) {
      if (select[_j] == false) continue;
      auto j = order[_j];
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(0.f, xx2 - xx1 + offset);
      auto h = std::max(0.f, yy2 - yy1 + offset);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold) select[_j] = false;
    }
  }
  return order_t.masked_select(select_t);
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_impl, CPU, nms_cpu);

Tensor softnms_cpu(Tensor boxes, Tensor scores, Tensor dets,
                   float iou_threshold, float sigma, float min_score,
                   int method, int offset) {
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }

  auto x1_t = boxes.select(1, 0).contiguous();
  auto y1_t = boxes.select(1, 1).contiguous();
  auto x2_t = boxes.select(1, 2).contiguous();
  auto y2_t = boxes.select(1, 3).contiguous();
  auto scores_t = scores.clone();

  Tensor areas_t = (x2_t - x1_t + offset) * (y2_t - y1_t + offset);

  auto nboxes = boxes.size(0);
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto sc = scores_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();
  auto de = dets.data_ptr<float>();

  int64_t pos = 0;
  Tensor inds_t = at::arange(nboxes, boxes.options().dtype(at::kLong));
  auto inds = inds_t.data_ptr<int64_t>();

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
    auto ix1 = de[i * 5 + 0] = x1[max_pos];
    auto iy1 = de[i * 5 + 1] = y1[max_pos];
    auto ix2 = de[i * 5 + 2] = x2[max_pos];
    auto iy2 = de[i * 5 + 3] = y2[max_pos];
    auto iscore = de[i * 5 + 4] = sc[max_pos];
    auto iarea = areas[max_pos];
    auto iind = inds[max_pos];
    x1[max_pos] = x1[i];
    y1[max_pos] = y1[i];
    x2[max_pos] = x2[i];
    y2[max_pos] = y2[i];
    sc[max_pos] = sc[i];
    areas[max_pos] = areas[i];
    inds[max_pos] = inds[i];
    x1[i] = ix1;
    y1[i] = iy1;
    x2[i] = ix2;
    y2[i] = iy2;
    sc[i] = iscore;
    areas[i] = iarea;
    inds[i] = iind;

    pos = i + 1;
    while (pos < nboxes) {
      auto xx1 = std::max(ix1, x1[pos]);
      auto yy1 = std::max(iy1, y1[pos]);
      auto xx2 = std::min(ix2, x2[pos]);
      auto yy2 = std::min(iy2, y2[pos]);

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
        x1[pos] = x1[nboxes - 1];
        y1[pos] = y1[nboxes - 1];
        x2[pos] = x2[nboxes - 1];
        y2[pos] = y2[nboxes - 1];
        sc[pos] = sc[nboxes - 1];
        areas[pos] = areas[nboxes - 1];
        inds[pos] = inds[nboxes - 1];
        nboxes = nboxes - 1;
        pos = pos - 1;
      }
      pos = pos + 1;
    }
  }
  return inds_t.slice(0, 0, nboxes);
}

Tensor softnms_impl(Tensor boxes, Tensor scores, Tensor dets,
                    float iou_threshold, float sigma, float min_score,
                    int method, int offset);
REGISTER_DEVICE_IMPL(softnms_impl, CPU, softnms_cpu);

std::vector<std::vector<int> > nms_match_cpu(Tensor dets, float iou_threshold) {
  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();
  auto scores = dets.select(1, 4).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t =
      at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();

  std::vector<int> keep;
  std::vector<std::vector<int> > matched;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) continue;
    keep.push_back(i);
    std::vector<int> v_i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<float>(0), xx2 - xx1);
      auto h = std::max(static_cast<float>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= iou_threshold) {
        suppressed[j] = 1;
        v_i.push_back(j);
      }
    }
    matched.push_back(v_i);
  }
  for (size_t i = 0; i < keep.size(); i++)
    matched[i].insert(matched[i].begin(), keep[i]);
  return matched;
}

std::vector<std::vector<int> > nms_match_impl(Tensor dets, float iou_threshold);
REGISTER_DEVICE_IMPL(nms_match_impl, CPU, nms_match_cpu);

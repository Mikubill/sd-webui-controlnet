// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "nms_pytorch.h"

using namespace parrots;

// Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);
template <typename T>
void nms_parrots(T& ctx, const SSElement& attr,
                 const OperatorBase::in_list_t& ins,
                 OperatorBase::out_list_t& outs) {
  float iou_threshold;
  int offset;
  SSAttrs(attr)
      .get("iou_threshold", iou_threshold)
      .get("offset", offset)
      .done();
  at::Tensor boxes, scores;
  boxes = buildATensor(ctx, ins[0]);
  scores = buildATensor(ctx, ins[1]);
  auto out = nms(boxes, scores, iou_threshold, offset);
  updateDArray(ctx, out, outs[0]);
}

/*Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
 *                float sigma, float min_score, int method, int offset);*/
template <typename T>
void softnms_parrots(T& ctx, const SSElement& attr,
                     const OperatorBase::in_list_t& ins,
                     OperatorBase::out_list_t& outs) {
  float iou_threshold, sigma, min_score;
  int method, offset;
  SSAttrs(attr)
      .get("iou_threshold", iou_threshold)
      .get("sigma", sigma)
      .get("min_score", min_score)
      .get("method", method)
      .get("offset", offset)
      .done();
  at::Tensor boxes, scores, dets;
  boxes = buildATensor(ctx, ins[0]);
  scores = buildATensor(ctx, ins[1]);
  dets = buildATensor(ctx, ins[2]);
  auto out = softnms(boxes, scores, dets, iou_threshold, sigma, min_score,
                     method, offset);
  updateDArray(ctx, out, outs[0]);
}

// std::vector<std::vector<int> > nms_match(Tensor dets, float iou_threshold);
template <typename T>
void nms_match_parrots(T& ctx, const SSElement& attr,
                       const OperatorBase::in_list_t& ins,
                       OperatorBase::out_list_t& outs) {
  float iou_threshold;
  SSAttrs(attr).get("iou_threshold", iou_threshold).done();
  at::Tensor dets;
  dets = buildATensor(ctx, ins[0]);
  auto out = nms_match(dets, iou_threshold);
  int n = out.size(), m = 0;
  for (int i = 0; i < n; ++i)
    if (m < out[i].size()) m = out[i].size();
  auto options = torch::TensorOptions().dtype(at::kInt);
  auto tensor = torch::zeros({n, m}, options);
  for (int i = 0; i < n; i++)
    tensor.slice(0, i, i + 1) =
        torch::from_blob(out[i].data(), {out[i].size()}, options);
  updateDArray(ctx, tensor, outs[0]);
}

/*Tensor nms_rotated(const Tensor dets, const Tensor scores, const Tensor order,
 *                    const Tensor dets_sorted, const float iou_threshold,
 *                                       const int multi_label);*/
template <typename T>
void nms_rotated_parrots(T& ctx, const SSElement& attr,
                         const OperatorBase::in_list_t& ins,
                         OperatorBase::out_list_t& outs) {
  float iou_threshold;
  int multi_label;
  SSAttrs(attr)
      .get("iou_threshold", iou_threshold)
      .get("multi_label", multi_label)
      .done();
  at::Tensor dets, scores, order, dets_sorted;
  dets = buildATensor(ctx, ins[0]);
  scores = buildATensor(ctx, ins[1]);
  order = buildATensor(ctx, ins[2]);
  dets_sorted = buildATensor(ctx, ins[3]);
  auto out =
      nms_rotated(dets, scores, order, dets_sorted, iou_threshold, multi_label);
  updateDArray(ctx, out, outs[0]);
}

PARROTS_EXTENSION_REGISTER(nms)
    .attr("iou_threshold")
    .attr("offset")
    .input(2)
    .output(1)
    .apply(nms_parrots<HostContext>)
#ifdef MMCV_WITH_CUDA
    .apply(nms_parrots<CudaContext>)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(softnms)
    .attr("iou_threshold")
    .attr("sigma")
    .attr("min_score")
    .attr("method")
    .attr("offset")
    .input(3)
    .output(1)
    .apply(softnms_parrots<HostContext>)
#ifdef MMCV_WITH_CUDA
    .apply(softnms_parrots<CudaContext>)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(nms_match)
    .attr("iou_threshold")
    .input(1)
    .output(1)
    .apply(nms_match_parrots<HostContext>)
#ifdef MMCV_WITH_CUDA
    .apply(nms_match_parrots<CudaContext>)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(nms_rotated)
    .attr("multi_label")
    .attr("iou_threshold")
    .input(4)
    .output(1)
    .apply(nms_rotated_parrots<HostContext>)
#ifdef MMCV_WITH_CUDA
    .apply(nms_rotated_parrots<CudaContext>)
#endif
    .done();

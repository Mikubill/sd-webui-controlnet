#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);

void bbox_overlaps_npu(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                       const int mode, const bool aligned, const int offset) {
  string modeStr = "iou";
  if (mode == 1) {
    modeStr = "iof";
  }
  float offset_ = 1;
  if (offset == 0) {
    offset_ = 0.01;
  }
  at::Tensor bboxes = at::ones_like(bboxes2);
  at::Tensor gtboxes = at::ones_like(bboxes1);
  bboxes = aligned ? bboxes2.transpose(0, 1) : bboxes2;
  gtboxes = aligned ? bboxes1.transpose(0, 1) : bboxes1;
  OpCommand cmd;
  cmd.Name("Iou")
      .Input(bboxes)
      .Input(gtboxes)
      .Output(ious)
      .Attr("mode", modeStr)
      .Attr("eps", offset_)
      .Attr("aligned", aligned)
      .Run();
}

REGISTER_NPU_IMPL(bbox_overlaps_impl, bbox_overlaps_npu);

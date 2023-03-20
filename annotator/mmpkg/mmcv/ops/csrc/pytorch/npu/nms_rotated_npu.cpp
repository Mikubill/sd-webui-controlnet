#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;

Tensor nms_rotated_npu(const Tensor dets, const Tensor scores,
                       const Tensor labels, const float iou_threshold) {
  auto originDtype = dets.scalar_type();
  at::Tensor detsCast = dets;
  at::Tensor scoresCast = scores;
  if (originDtype != at::ScalarType::Float) {
    detsCast = NPUNativeFunctions::npu_dtype_cast(dets, at::kFloat);
    scoresCast = NPUNativeFunctions::npu_dtype_cast(scores, at::kFloat);
  }
  c10::SmallVector<int64_t, SIZE> selectedIndexSize = {dets.size(0)};
  at::Tensor selectedBox = OpPreparation::ApplyTensor(dets);
  at::Tensor selectedIndex = OpPreparation::ApplyTensor(
      selectedIndexSize, dets.options().dtype(at::kInt), dets);

  c10::SmallVector<int64_t, N> output_sync_idx = {0, 1};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("RotatedNMS")
      .Input(detsCast)
      .Input(scoresCast)
      .Input(labels)
      .Output(selectedBox)
      .Output(selectedIndex)
      .Attr("iou_threshold", (float)iou_threshold)
      .Run();
  selectedIndex = NPUNativeFunctions::npu_dtype_cast(selectedIndex, at::kLong);
  return selectedIndex;
}

#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

Tensor nms_npu(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  int64_t offset_64 = offset;
  at::Tensor iou_threshold_y = at_npu::native::OpPreparation::ApplyTensor(
                                   {}, boxes.options().dtype(at::kFloat), boxes)
                                   .fill_(iou_threshold);
  at::Tensor scores_threshold_y =
      at_npu::native::OpPreparation::ApplyTensor(
          {}, boxes.options().dtype(at::kFloat), boxes)
          .fill_(0);
  at::Tensor max_outputsize_y = at_npu::native::OpPreparation::ApplyTensor(
                                    {}, boxes.options().dtype(at::kInt), boxes)
                                    .fill_(boxes.size(0));
  c10::SmallVector<int64_t, SIZE> outputsize = {boxes.size(0)};
  at::Tensor output = at_npu::native::OpPreparation::ApplyTensor(
                          outputsize, boxes.options().dtype(at::kInt), boxes)
                          .fill_(-1);
  OpCommand cmd;
  cmd.Name("NonMaxSuppressionV3")
      .Input(boxes)
      .Input(scores)
      .Input(max_outputsize_y)
      .Input(iou_threshold_y)
      .Input(scores_threshold_y)
      .Attr("offset", offset_64)
      .Output(output)
      .Run();
  auto outputsizeBool = at::gt(output, -1);
  auto outputsizeInt = outputsizeBool.to(at::ScalarType::Int);
  auto countLen = at::sum(outputsizeInt, at::ScalarType::Int);
  at::Tensor actual_output = output.slice(0, 0, countLen.item().toLong());
  actual_output = at_npu::native::NPUNativeFunctions::npu_dtype_cast(
      actual_output, at::kLong);
  return actual_output;
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);

REGISTER_NPU_IMPL(nms_impl, nms_npu);

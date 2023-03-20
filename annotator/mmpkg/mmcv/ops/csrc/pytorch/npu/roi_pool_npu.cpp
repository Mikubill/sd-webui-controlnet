#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void roi_pool_forward_npu(Tensor input, Tensor rois, Tensor output,
                          Tensor argmax, int pooled_height, int pooled_width,
                          float spatial_scale) {
  int64_t pooled_height_64 = pooled_height;
  int64_t pooled_width_64 = pooled_width;
  int64_t pooled_channel = 1;
  at::Tensor roi_actual_num = at_npu::native::OpPreparation::ApplyTensor(
      {}, rois.options().dtype(at::kInt), rois);
  OpCommand cmd;
  cmd.Name("RoiPoolingWithArgMax")
      .Input(input)
      .Input(rois)
      .Input(roi_actual_num)
      .Output(output)
      .Output(argmax)
      .Attr("pooled_h", pooled_height_64)
      .Attr("pooled_w", pooled_width_64)
      .Attr("spatial_scale_h", spatial_scale)
      .Attr("spatial_scale_w", spatial_scale)
      .Attr("pool_channel", pooled_channel)
      .Run();
}

void roi_pool_backward_npu(Tensor grad_output, Tensor rois, Tensor argmax,
                           Tensor grad_input, int pooled_height,
                           int pooled_width, float spatial_scale) {
  int64_t pooled_height_64 = pooled_height;
  int64_t pooled_width_64 = pooled_width;
  int64_t pooled_channel = 1;
  at::Tensor roi_actual_num = at_npu::native::OpPreparation::ApplyTensor(
      {}, rois.options().dtype(at::kInt), rois);
  at::Tensor x = at::ones_like(grad_input);
  OpCommand cmd;
  cmd.Name("RoiPoolingGradWithArgMax")
      .Input(grad_output)
      .Input(x)
      .Input(rois)
      .Input(roi_actual_num)
      .Input(argmax)
      .Output(grad_input)
      .Attr("pooled_h", pooled_height_64)
      .Attr("pooled_w", pooled_width_64)
      .Attr("spatial_scale_h", spatial_scale)
      .Attr("spatial_scale_w", spatial_scale)
      .Attr("pool_channel", pooled_channel)
      .Run();
}

void roi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale);

void roi_pool_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale);

REGISTER_NPU_IMPL(roi_pool_forward_impl, roi_pool_forward_npu);
REGISTER_NPU_IMPL(roi_pool_backward_impl, roi_pool_backward_npu);

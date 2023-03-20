#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void deform_roi_pool_forward_impl(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma);

void deform_roi_pool_backward_impl(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma);

void deform_roi_pool_forward_npu(Tensor input, Tensor rois, Tensor offset,
                                 Tensor output, int pooled_height,
                                 int pooled_width, float spatial_scale,
                                 int sampling_ratio, float gamma) {
  c10::SmallVector<int64_t, 2> output_sizes = {pooled_height, pooled_width};
  at::IntArrayRef output_size = at::IntArrayRef(output_sizes);
  int64_t sampling_ratio_ = (int64_t)sampling_ratio;
  OpCommand cmd;
  cmd.Name("DeformableRoiPool")
      .Input(input)
      .Input(rois)
      .Input(offset)
      .Output(output)
      .Attr("spatial_scale", spatial_scale)
      .Attr("output_size", output_size)
      .Attr("sampling_ratio", sampling_ratio_)
      .Attr("gamma", gamma)
      .Run();
}

void deform_roi_pool_backward_npu(Tensor grad_output, Tensor input, Tensor rois,
                                  Tensor offset, Tensor grad_input,
                                  Tensor grad_offset, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  c10::SmallVector<int64_t, 2> output_sizes = {pooled_height, pooled_width};
  at::IntArrayRef output_size = at::IntArrayRef(output_sizes);
  int64_t sampling_ratio_ = (int64_t)sampling_ratio;
  OpCommand cmd;
  cmd.Name("DeformableRoiPoolGrad")
      .Input(grad_output)
      .Input(input)
      .Input(rois)
      .Input(offset)
      .Output(grad_input)
      .Output(grad_offset)
      .Attr("output_size", output_size)
      .Attr("spatial_scale", spatial_scale)
      .Attr("sample_ratio", sampling_ratio_)
      .Attr("gamma", gamma)
      .Run();
}

REGISTER_NPU_IMPL(deform_roi_pool_forward_impl, deform_roi_pool_forward_npu);

REGISTER_NPU_IMPL(deform_roi_pool_backward_impl, deform_roi_pool_backward_npu);

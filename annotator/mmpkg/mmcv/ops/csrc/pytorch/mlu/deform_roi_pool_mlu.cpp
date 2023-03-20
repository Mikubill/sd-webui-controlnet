/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

void KernelDeformRoIPoolForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                cnrtQueue_t queue, cnrtDataType_t data_type,
                                const void *input, const void *rois,
                                const void *offset, void *output,
                                const int channels, const int height,
                                const int width, const int num_rois,
                                const int pooled_height, const int pooled_width,
                                const float spatial_scale,
                                const int sampling_ratio, const float gamma);

void KernelDeformRoIPoolBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    cnrtDataType_t data_type, const void *grad_output, const void *input,
    const void *rois, const void *offset, void *grad_input, void *grad_offset,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, const float spatial_scale,
    const int sampling_ratio, const float gamma);

// policy function for forward and backward
static void policyFunc(const int bin_num, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  const size_t cluster_limit = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  ;
  const size_t core_limit = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  const size_t bin_num_align = CEIL_ALIGN(bin_num, core_limit);
  k_dim->x = core_limit;
  k_dim->y = (bin_num_align / core_limit) > cluster_limit
                 ? cluster_limit
                 : (bin_num_align / core_limit);
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

void DeformRoIPoolForwardMLUKernelLauncher(Tensor input, Tensor rois,
                                           Tensor offset, Tensor output,
                                           int pooled_height, int pooled_width,
                                           float spatial_scale,
                                           int sampling_ratio, float gamma) {
  // Check dtype.
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "input type should be Float or Half, got ", input.scalar_type());
  TORCH_CHECK(input.scalar_type() == rois.scalar_type(),
              "rois should have the same type as input");

  // Check shape.
  TORCH_CHECK(input.dim() == 4, "input should be 4d tensor, got ", input.dim(),
              "D.");
  TORCH_CHECK(rois.dim() == 2, "rois should be 2d tensor, got ", rois.dim(),
              "D.");
  if (offset.defined() && offset.numel() > 0) {
    TORCH_CHECK(input.scalar_type() == offset.scalar_type(),
                "offset should have the same type as input");
    TORCH_CHECK(offset.dim() == 4, "offset should be 4d tensor, got ",
                offset.dim(), "D.");
    TORCH_CHECK(
        (offset.size(0) == rois.size(0)), "offset.size(0) = ", offset.size(0),
        "while rois.size(0)) = ", rois.size(0), ". They should be the same.");
    TORCH_CHECK((offset.size(1) == 2), "offset.size(1) should be 2, ",
                "but now offset.size(1) = ", offset.size(1), ".");
    TORCH_CHECK((offset.size(2) == output.size(2)),
                "offset.size(2) = ", offset.size(2),
                "while output.size(2)) = ", output.size(2),
                ". They should be the same.");
    TORCH_CHECK((offset.size(3) == output.size(3)),
                "offset.size(3) = ", offset.size(3),
                "while output.size(3)) = ", output.size(3),
                ". They should be the same.");
  }

  TORCH_CHECK(spatial_scale > 0 && spatial_scale <= 1,
              "spatial_scale should be within (0, 1], got ", spatial_scale,
              ".");

  // compute kernel params
  auto height = input.size(2);
  auto width = input.size(3);
  auto channels = input.size(1);
  auto num_rois = output.size(0);

  if (output.numel() == 0) {
    output = at::zeros({num_rois, channels, pooled_height, pooled_width},
                       input.options());
    return;
  }

  // zero element check
  TORCH_CHECK(input.size(0) != 0, "input.size(0) should not be zero, got ",
              input.size(0));
  TORCH_CHECK(rois.numel() != 0, "rois.numel() should not be zero, got ",
              rois.numel());
  if (input.numel() == 0 || output.numel() == 0) {
    return;
  }

  // large tensor check
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  TORCH_CHECK(input.numel() < max_input_num,
              "input.numel() should be less than 2147483648, got ",
              input.numel());
  TORCH_CHECK(rois.numel() < max_input_num,
              "rois.numel() should be less than 2147483648, got ",
              rois.numel());
  TORCH_CHECK(output.numel() < max_input_num,
              "output.numel() should be less than 2147483648, got ",
              output.numel());
  TORCH_CHECK(!offset.defined() || offset.numel() < max_input_num,
              "offset.numel() should be less than 2147483648, got ",
              offset.numel());

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_ = torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);

  at::Tensor output_ =
      at::empty({num_rois, channels, pooled_height, pooled_width},
                input.options(), memory_format);

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(num_rois * pooled_height * pooled_width, &k_dim, &k_type);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto offset_impl = torch_mlu::getMluTensorImpl(offset);
  auto offset_ptr = offset_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output_);
  auto output_ptr = output_impl->cnnlMalloc();

  // get comput dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(input_.dtype());

  // launch kernel
  CNLOG(INFO) << "Launch Kernel MLUKernelDeformRoIPoolForward<<<" << k_dim.x
              << ", " << k_dim.y << ", " << k_dim.z << ">>>";

  KernelDeformRoIPoolForward(k_dim, k_type, queue, data_type, input_ptr,
                             rois_ptr, offset_ptr, output_ptr, channels, height,
                             width, num_rois, pooled_height, pooled_width,
                             spatial_scale, sampling_ratio, gamma);

  output.copy_(output_);
}

void DeformRoIPoolBackwardMLUKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor offset,
    Tensor grad_input, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma) {
  // Check dtype.
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "input type should be Float or Half, got ", input.scalar_type());
  TORCH_CHECK(input.scalar_type() == grad_output.scalar_type(),
              "grad_output should have the same type as input");
  TORCH_CHECK(input.scalar_type() == rois.scalar_type(),
              "rois should have the same type as input");
  TORCH_CHECK(input.scalar_type() == grad_input.scalar_type(),
              "grad_input should have the same type as input");

  // Check shape.
  TORCH_CHECK(grad_output.dim() == 4, "grad_output should be 4d tensor, got ",
              grad_output.dim(), "D.");
  TORCH_CHECK(input.dim() == 4, "input should be 4d tensor, got ", input.dim(),
              "D.");
  TORCH_CHECK(rois.dim() == 2, "rois should be 2d tensor, got ", rois.dim(),
              "D.");
  if (offset.defined() && offset.numel() > 0) {
    TORCH_CHECK(input.scalar_type() == offset.scalar_type(),
                "offset should have the same type as input");
    TORCH_CHECK(offset.dim() == 4, "offset should be 4d tensor, got ",
                offset.dim(), "D.");
    TORCH_CHECK(
        (offset.size(0) == rois.size(0)), "offset.size(0) = ", offset.size(0),
        "while rois.size(0)) = ", rois.size(0), ". They should be the same.");
    TORCH_CHECK((offset.size(1) == 2), "offset.size(1) should be 2, ",
                "but now offset.size(1) = ", offset.size(1), ".");
    TORCH_CHECK((offset.size(2) == grad_output.size(2)),
                "offset.size(2) = ", offset.size(2),
                "while grad_output.size(2)) = ", grad_output.size(2),
                ". They should be the same.");
    TORCH_CHECK((offset.size(3) == grad_output.size(3)),
                "offset.size(3) = ", offset.size(3),
                "while grad_output.size(3)) = ", grad_output.size(3),
                ". They should be the same.");
  }

  TORCH_CHECK(spatial_scale > 0 && spatial_scale <= 1,
              "spatial_scale should be within (0, 1], got ", spatial_scale);

  // Check relationship between tensor.
  TORCH_CHECK((grad_output.size(0) == rois.size(0)),
              "grad_output.size(0) = ", grad_output.size(0),
              "while rois.size(0)) = ", rois.size(0),
              ". They should be the same.");
  TORCH_CHECK((grad_output.size(1) == input.size(1)),
              "grad_output.size(1) = ", grad_output.size(1),
              "while input.size(1)) = ", input.size(1),
              ". They should be the same.");
  TORCH_CHECK((grad_output.size(2) == pooled_height),
              "grad_output.size(2) = ", grad_output.size(2),
              "while pooled_height = ", pooled_height,
              ". They should be the same.");
  TORCH_CHECK((grad_output.size(3) == pooled_width),
              "grad_output.size(3) = ", grad_output.size(3),
              "while pooled_width = ", pooled_width,
              ". They should be the same.");

  // compute kernel params
  auto batch = input.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);
  auto num_rois = grad_output.size(0);

  // zero element check
  TORCH_CHECK(input.size(0) != 0, "input.size(0) should not be zero, got ",
              input.size(0));
  TORCH_CHECK(rois.numel() != 0, "rois.numel() should not be zero, got ",
              rois.numel());
  if (input.numel() == 0 || grad_output.numel() == 0) {
    return;
  }

  // large tensor check
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  TORCH_CHECK(input.numel() < max_input_num,
              "input.numel() should be less than 2147483648, got ",
              input.numel());
  TORCH_CHECK(rois.numel() < max_input_num,
              "rois.numel() should be less than 2147483648, got ",
              rois.numel());
  TORCH_CHECK(grad_output.numel() < max_input_num,
              "grad_output.numel() should be less than 2147483648, got ",
              grad_output.numel());
  TORCH_CHECK(!offset.defined() || offset.numel() < max_input_num,
              "offset.numel() should be less than 2147483648, got ",
              offset.numel());

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_output.dim());
  auto grad_output_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(grad_output, memory_format);
  memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_ = torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);
  at::Tensor grad_input_ = at::empty({batch, channels, height, width},
                                     input.options(), memory_format)
                               .zero_();

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(num_rois * pooled_height * pooled_width, &k_dim, &k_type);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto grad_output_impl = torch_mlu::getMluTensorImpl(grad_output_);
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto input_impl = torch_mlu::getMluTensorImpl(input_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto offset_impl = torch_mlu::getMluTensorImpl(offset);
  auto offset_ptr = offset_impl->cnnlMalloc();
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input_);
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  auto grad_offset_impl = torch_mlu::getMluTensorImpl(grad_offset);
  auto grad_offset_ptr = grad_offset_impl->cnnlMalloc();

  // get comput dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(input.dtype());

  // launch kernel
  CNLOG(INFO) << "Launch Kernel KernelDeformRoIPoolBackward<<<" << k_dim.x
              << ", " << k_dim.y << ", " << k_dim.z << ">>>";

  KernelDeformRoIPoolBackward(k_dim, k_type, queue, data_type, grad_output_ptr,
                              input_ptr, rois_ptr, offset_ptr, grad_input_ptr,
                              grad_offset_ptr, channels, height, width,
                              num_rois, pooled_height, pooled_width,
                              spatial_scale, sampling_ratio, gamma);

  grad_input.copy_(grad_input_);
}

void deform_roi_pool_forward_mlu(Tensor input, Tensor rois, Tensor offset,
                                 Tensor output, int pooled_height,
                                 int pooled_width, float spatial_scale,
                                 int sampling_ratio, float gamma) {
  DeformRoIPoolForwardMLUKernelLauncher(input, rois, offset, output,
                                        pooled_height, pooled_width,
                                        spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_backward_mlu(Tensor grad_output, Tensor input, Tensor rois,
                                  Tensor offset, Tensor grad_input,
                                  Tensor grad_offset, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  DeformRoIPoolBackwardMLUKernelLauncher(
      grad_output, input, rois, offset, grad_input, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma);
}

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

REGISTER_DEVICE_IMPL(deform_roi_pool_forward_impl, MLU,
                     deform_roi_pool_forward_mlu);
REGISTER_DEVICE_IMPL(deform_roi_pool_backward_impl, MLU,
                     deform_roi_pool_backward_mlu);

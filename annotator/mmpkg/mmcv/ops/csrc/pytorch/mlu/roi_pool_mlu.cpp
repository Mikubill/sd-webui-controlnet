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

void KernelRoiPoolForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                          cnrtQueue_t queue, cnrtDataType_t data_type,
                          const void *input_data, const void *input_rois,
                          const int batch, const int channels, const int height,
                          const int width, const int pooled_height,
                          const int pooled_width, const int rois_num,
                          const float spatial_scale, void *output_data,
                          int *argmax);

void KernelRoiPoolBackward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                           cnrtQueue_t queue, cnrtDataType_t k_dtype,
                           const void *grad_output_ptr, const void *rois_ptr,
                           const int *argmax_ptr, void *grad_input_ptr,
                           const int box_num, const int pooled_height,
                           const int pooled_width, const int channels,
                           const int batch, const int height, const int width,
                           const float spatial_scale);

// policy function for forward
static void policyFuncForward(const int bin_num, cnrtDim3_t *k_dim,
                              cnrtFunctionType_t *k_type) {
  auto core_num = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_num;
  unsigned int use_cluster = bin_num / core_num + (bin_num % core_num > 0);
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

void ROIPoolForwardMLUKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                     Tensor argmax, int pooled_height,
                                     int pooled_width, float spatial_scale) {
  // Check dtype.
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "input type should be Float or Half, got ", input.scalar_type());
  TORCH_CHECK(input.scalar_type() == rois.scalar_type(),
              "rois should have the same type as input");

  // Check dtype relationship.
  TORCH_CHECK(
      argmax.scalar_type() == at::kLong || argmax.scalar_type() == at::kInt,
      "argmax type should be Int or Long, got ", argmax.scalar_type());

  // Check shape.
  TORCH_CHECK(input.dim() == 4, "input should be 4d tensor, got ", input.dim(),
              "D");
  TORCH_CHECK(rois.dim() == 2, "rois should be 2d tensor, got ", rois.dim(),
              "D");
  TORCH_CHECK(argmax.dim() == 4, "argmax should be 4d tensor, got ",
              argmax.dim(), "D");

  TORCH_CHECK(spatial_scale > 0 && spatial_scale <= 1,
              "spatial_scale should be within (0, 1], got ", spatial_scale);

  // compute kernel params
  auto batch = input.size(0);
  auto height = input.size(2);
  auto width = input.size(3);
  auto channels = input.size(1);
  auto rois_num = output.size(0);

  if (output.numel() == 0) {
    output = at::zeros({rois_num, channels, pooled_height, pooled_width},
                       input.options());
    return;
  }
  if (argmax.numel() == 0) {
    argmax = at::zeros({rois_num, channels, pooled_height, pooled_width},
                       argmax.options());
    return;
  }

  // zero element check
  if (input.numel() == 0 || rois.numel() == 0 || output.numel() == 0 ||
      argmax.numel() == 0) {
    return;
  }

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_ = torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);

  at::Tensor output_ =
      at::empty({rois_num, channels, pooled_height, pooled_width},
                input.options(), memory_format);
  at::Tensor argmax_ =
      at::empty({rois_num, channels, pooled_height, pooled_width},
                argmax.options(), memory_format);

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncForward(rois_num * pooled_height * pooled_width, &k_dim, &k_type);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output_);
  auto output_ptr = output_impl->cnnlMalloc();
  auto argmax_impl = torch_mlu::getMluTensorImpl(argmax_);
  auto argmax_ptr = argmax_impl->cnnlMalloc();

  // get comput dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(input_.dtype());

  // launch kernel
  CNLOG(INFO) << "Launch Kernel MLUKernelRoiPoolForward<<<" << k_dim.x << ", "
              << k_dim.y << ", " << k_dim.z << ">>>";

  KernelRoiPoolForward(k_dim, k_type, queue, data_type, input_ptr, rois_ptr,
                       batch, channels, height, width, pooled_height,
                       pooled_width, rois_num, spatial_scale, output_ptr,
                       (int *)argmax_ptr);
  output.copy_(output_);
  argmax.copy_(argmax_);
}

// policy function for backward
static void policyFuncBackward(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  k_dim->z = 1;
}

void ROIPoolBackwardMLUKernelLauncher(Tensor grad_output, Tensor rois,
                                      Tensor argmax, Tensor grad_input,
                                      int pooled_height, int pooled_width,
                                      float spatial_scale) {
  // Check dtype.
  TORCH_CHECK(
      argmax.scalar_type() == at::kLong || argmax.scalar_type() == at::kInt,
      "argmax type should be Int or Long, got ", argmax.scalar_type());
  TORCH_CHECK((grad_output.scalar_type() == at::kFloat ||
               grad_output.scalar_type() == at::kHalf),
              "grad_output type should be FLoat or Half, got ",
              grad_output.scalar_type());

  // Check dtype relationship.
  TORCH_CHECK((rois.scalar_type() == grad_output.scalar_type()),
              "rois should have the same type as grad_output");

  // Check shape.
  TORCH_CHECK(grad_output.dim() == 4, "grad_output should be 4d tensor, got ",
              grad_output.dim(), "D");
  TORCH_CHECK(rois.dim() == 2, "rois should be 2d tensor, got ", rois.dim(),
              "D");
  TORCH_CHECK(argmax.dim() == 4, "argmax should be 4d tensor, got ",
              argmax.dim(), "D");

  TORCH_CHECK(spatial_scale > 0 && spatial_scale <= 1,
              "spatial_scale should be within (0, 1], got ", spatial_scale);

  // Check relationship between tensor.
  // Check the relationship of n.
  TORCH_CHECK(grad_output.size(0) == rois.size(0),
              "grad_output.size(0) = ", grad_output.size(0),
              ", while rois.size(0) = ", rois.size(0),
              ". They should be the same.");

  // Check the relationship of channels.
  TORCH_CHECK(grad_output.size(1) == argmax.size(1),
              "grad_output.size(1) = ", grad_output.size(1),
              ", while argmax.size(1) = ", argmax.size(1),
              ". They should be the same.");

  // Check the relationship of height and width.
  TORCH_CHECK(grad_output.size(2) == argmax.size(2),
              "argmax.size(2) = ", argmax.size(2),
              ", while grad_output.size(2) = ", grad_output.size(2),
              ". They should be the same.");
  TORCH_CHECK(grad_output.size(3) == argmax.size(3),
              "argmax.size(3) = ", argmax.size(3),
              ", while grad_output.size(3) = ", grad_output.size(3),
              ". They should be the same.");

  // Check zero element.
  if (grad_output.numel() == 0 || rois.numel() == 0 || argmax.numel() == 0 ||
      grad_input.numel() == 0) {
    // return if zero-element
    return;
  }

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_output.dim());
  auto grad_output_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(grad_output, memory_format);
  auto argmax_ = torch_mlu::cnnl::ops::cnnl_contiguous(argmax, memory_format);

  int boxes_num = grad_output.size(0);
  int no = grad_input.size(0);
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);
  auto grad_input_ = at::empty({no, channels, height, width},
                               grad_input.options(), memory_format)
                         .zero_();

  // get tensor impl
  auto grad_output_impl = torch_mlu::getMluTensorImpl(grad_output_);
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto argmax_impl = torch_mlu::getMluTensorImpl(argmax_);
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input_);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get mlu ptr
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto argmax_ptr = argmax_impl->cnnlMalloc();
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();

  // calculate task dimension
  cnrtDataType_t k_dtype = torch_mlu::toCnrtDtype(grad_input.dtype());
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncBackward(&k_dim, &k_type);

  CNLOG(INFO) << "Launch Kernel MLUKernelRoiPoolBackward<<<" << k_dim.x << ", "
              << k_dim.y << ", " << k_dim.z << ">>>";

  KernelRoiPoolBackward(k_dim, k_type, queue, k_dtype, grad_output_ptr,
                        rois_ptr, (int *)argmax_ptr, grad_input_ptr, boxes_num,
                        pooled_height, pooled_width, channels, no, height,
                        width, spatial_scale);

  grad_input.copy_(grad_input_);
}

void roi_pool_forward_mlu(Tensor input, Tensor rois, Tensor output,
                          Tensor argmax, int pooled_height, int pooled_width,
                          float spatial_scale) {
  ROIPoolForwardMLUKernelLauncher(input, rois, output, argmax, pooled_height,
                                  pooled_width, spatial_scale);
}

void roi_pool_backward_mlu(Tensor grad_output, Tensor rois, Tensor argmax,
                           Tensor grad_input, int pooled_height,
                           int pooled_width, float spatial_scale) {
  ROIPoolBackwardMLUKernelLauncher(grad_output, rois, argmax, grad_input,
                                   pooled_height, pooled_width, spatial_scale);
}

void roi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale);

void roi_pool_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale);

REGISTER_DEVICE_IMPL(roi_pool_forward_impl, MLU, roi_pool_forward_mlu);
REGISTER_DEVICE_IMPL(roi_pool_backward_impl, MLU, roi_pool_backward_mlu);

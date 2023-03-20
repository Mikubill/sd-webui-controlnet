/*************************************************************************
 * Copyright (C) 2021 Cambricon.
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

void KernelRoiAlign(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                    cnrtQueue_t queue, const cnrtDataType_t d_type,
                    const void *input, const void *rois, const int channels,
                    const bool aligned, const int pooled_height,
                    const int pooled_width, const int input_height,
                    const int input_width, const int sampling_ratio,
                    const float spatial_scale, const int num_rois,
                    void *output);

void KernelRoiAlignBackward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                            cnrtQueue_t queue, const cnrtDataType_t dtype,
                            const void *grads, const void *boxes,
                            void *grads_image, const int boxes_num,
                            const int hi, const int wi, const int c,
                            const int no, const int ho, const int wo,
                            const float spatial_scale, const int sampling_ratio,
                            const bool aligned);

void ROIAlignForwardMLUKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax_y, Tensor argmax_x,
                                      int aligned_height, int aligned_width,
                                      float spatial_scale, int sampling_ratio,
                                      int pool_mode, bool aligned) {
  // params check
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "input type should be Float or Half, got ", input.scalar_type());
  TORCH_CHECK(rois.scalar_type() == input.scalar_type(),
              "rois should have the same type as input");
  TORCH_CHECK(input.dim() == 4, "input should be a 4d tensor, got ",
              input.dim(), "D");
  TORCH_CHECK(rois.dim() == 2, "rois should be a 2d tensor, got ", rois.dim(),
              "D");
  TORCH_CHECK(pool_mode == 1, "pool_mode only supports 'avg' currently");

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_tensor =
      torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  if (output.numel() == 0) {
    output = at::zeros({num_rois, channels, aligned_height, aligned_width},
                       input.options());
    return;
  }

  at::Tensor output_tmp =
      at::empty({num_rois, channels, aligned_height, aligned_width},
                input.options(), memory_format);

  // get tensor impl
  auto self_impl = torch_mlu::getMluTensorImpl(input_tensor);
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto output_impl = torch_mlu::getMluTensorImpl(output_tmp);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get the mlu ptr
  auto self_ptr = self_impl->cnnlMalloc();
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  k_dim.x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.y = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  k_dim.z = 1;
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(input.dtype());

  KernelRoiAlign(k_dim, k_type, queue, data_type, self_ptr, rois_ptr, channels,
                 aligned, aligned_height, aligned_width, height, width,
                 sampling_ratio, spatial_scale, num_rois, output_ptr);

  output.copy_(output_tmp);
}

static int nearestPower2(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

void ROIAlignBackwardMLUKernelLauncher(Tensor grad, Tensor rois,
                                       Tensor argmax_y, Tensor argmax_x,
                                       Tensor grad_input, int aligned_height,
                                       int aligned_width, float spatial_scale,
                                       int sampling_ratio, int pool_mode,
                                       bool aligned) {
  // params check
  TORCH_CHECK(
      grad.scalar_type() == at::kFloat || grad.scalar_type() == at::kHalf,
      "grad type should be Float or Half, got ", grad.scalar_type());
  TORCH_CHECK(rois.scalar_type() == grad.scalar_type(),
              "rois should have the same type as grad");
  TORCH_CHECK(grad.dim() == 4, "grad should be a 4d tensor, got ", grad.dim(),
              "D");
  TORCH_CHECK(rois.dim() == 2, "rois should be a 2d tensor, got ", rois.dim(),
              "D");
  TORCH_CHECK(pool_mode == 1, "pool_mode only supports 'avg' currently");

  int batch_size = grad_input.size(0);
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad.dim());
  auto grad_ = torch_mlu::cnnl::ops::cnnl_contiguous(grad, memory_format);
  auto grad_input_ = at::empty({batch_size, channels, height, width},
                               grad.options(), memory_format)
                         .zero_();

  int boxes_num = rois.size(0);
  int hi = grad.size(2);
  int wi = grad.size(3);
  int c = grad.size(1);

  int no = grad_input.size(0);
  int ho = grad_input.size(2);
  int wo = grad_input.size(3);

  // get tensor impl
  auto grad_impl = torch_mlu::getMluTensorImpl(grad_);
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input_);
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get the mlu ptr
  auto grad_ptr = grad_impl->cnnlMalloc();
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();

  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  int need_core = nearestPower2(boxes_num);
  int union_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t dim_x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  uint32_t dim_y = (need_core - 1) / dim_x + 1;
  dim_y = (dim_y > union_number) ? union_number : dim_y;
  cnrtDim3_t k_dim = {dim_x, dim_y, 1};
  cnrtDataType_t k_dtype = torch_mlu::toCnrtDtype(grad.dtype());

  KernelRoiAlignBackward(k_dim, k_type, queue, k_dtype, grad_ptr, rois_ptr,
                         grad_input_ptr, boxes_num, hi, wi, c, no, ho, wo,
                         spatial_scale, sampling_ratio, aligned);
  grad_input.copy_(grad_input_);
}

void roi_align_forward_mlu(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax_y, Tensor argmax_x, int aligned_height,
                           int aligned_width, float spatial_scale,
                           int sampling_ratio, int pool_mode, bool aligned) {
  ROIAlignForwardMLUKernelLauncher(input, rois, output, argmax_y, argmax_x,
                                   aligned_height, aligned_width, spatial_scale,
                                   sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_mlu(Tensor grad_output, Tensor rois, Tensor argmax_y,
                            Tensor argmax_x, Tensor grad_input,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  ROIAlignBackwardMLUKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned);

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned);

REGISTER_DEVICE_IMPL(roi_align_forward_impl, MLU, roi_align_forward_mlu);
REGISTER_DEVICE_IMPL(roi_align_backward_impl, MLU, roi_align_backward_mlu);

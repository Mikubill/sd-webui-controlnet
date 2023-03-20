/*************************************************************************
 * Copyright (C) 2022 by Cambricon.
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
#include "roi_align_rotated_utils.hpp"

namespace {

void policyFunc(int bin_num, cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  unsigned int core_num = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  unsigned int cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_num;
  unsigned int use_cluster = (bin_num + core_num - 1) / core_num;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

}  // namespace

void KernelRoiAlignRotatedForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const void *features, const void *rois,
    void *output, const int batch, const int height, const int width,
    const int channel, const int rois_num,
    const RoiAlignRotatedParams roiAlignRotatedParams);

void KernelRoiAlignRotatedBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const void *top_grad, const void *rois,
    void *bottom_grad, const int batch, const int height, const int width,
    const int channel, const int rois_num,
    const RoiAlignRotatedParams roiAlignRotatedParams);

void ROIAlignRotatedForwardMLUKernelLauncher(Tensor input, Tensor rois,
                                             Tensor output, int pooled_height,
                                             int pooled_width,
                                             float spatial_scale,
                                             int sampling_ratio, bool aligned,
                                             bool clockwise) {
  TORCH_CHECK(((input.scalar_type() == output.scalar_type()) &&
               (output.scalar_type() == rois.scalar_type())),
              "data types of input, rois and output should be the same, ",
              "but now input type is ", input.scalar_type(), ", rois type is ",
              rois.scalar_type(), ", output type is ", output.scalar_type(),
              ".");
  TORCH_CHECK(
      (input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf),
      "input type should be Float or Half, got ", input.scalar_type(), ".");

  TORCH_CHECK(input.dim() == 4, "input should be a 4d tensor, got ",
              input.dim(), "D.");
  TORCH_CHECK(rois.dim() == 2, "rois should be a 2d tensor, got ", rois.dim(),
              "D.");
  TORCH_CHECK(output.dim() == 4, "output should be a 4d tensor, got ",
              output.dim(), "D.");

  TORCH_CHECK((rois.size(0) == output.size(0)),
              "the 1st dimensions of rois and output should be the same, ",
              "but now the 1st dimension of rois is ", rois.size(0),
              ", and output is ", output.size(0), ".");

  TORCH_CHECK((input.size(1) == output.size(1)),
              "the 2nd dimensions of input and output should be the same, ",
              "but now the 2nd dimension of input is ", input.size(1),
              ", and output is ", output.size(1), ".");

  int channel = input.size(1);
  int width = input.size(3);
  int height = input.size(2);
  int batch = input.size(0);
  int rois_nums = rois.size(0);
  cnrtDataType_t d_type = torch_mlu::toCnrtDtype(input.dtype());

  // return if zero-elements
  if (input.numel() == 0) {
    CNLOG(INFO) << "Skip the zero-elements case.";
    return;
  }

  RoiAlignRotatedParams roiAlignRotatedParams{pooled_height,  pooled_width,
                                              sampling_ratio, spatial_scale,
                                              aligned,        clockwise};
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(rois_nums * pooled_height * pooled_width, &k_dim, &k_type);

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_tensor =
      torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);
  at::Tensor output_tmp =
      at::empty({rois_nums, channel, pooled_height, pooled_width},
                input.options(), memory_format);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_tensor);
  auto input_ptr = input_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output_tmp);
  auto output_ptr = output_impl->cnnlMalloc();

  KernelRoiAlignRotatedForward(k_dim, k_type, queue, d_type, input_ptr,
                               rois_ptr, output_ptr, batch, height, width,
                               channel, rois_nums, roiAlignRotatedParams);
  output.copy_(output_tmp);
}

void ROIAlignRotatedBackwardMLUKernelLauncher(
    Tensor top_grad, Tensor rois, Tensor bottom_grad, int pooled_height,
    int pooled_width, float spatial_scale, int sampling_ratio, bool aligned,
    bool clockwise) {
  TORCH_CHECK(((top_grad.scalar_type() == bottom_grad.scalar_type()) &&
               (bottom_grad.scalar_type() == rois.scalar_type())),
              "data types of top_grad, rois and bottom_grad should be ",
              "the same, but now top_grad type is ", top_grad.scalar_type(),
              ", rois type is ", rois.scalar_type(), ", bottom_grad type is ",
              bottom_grad.scalar_type(), ".");
  TORCH_CHECK((bottom_grad.scalar_type() == at::kFloat ||
               bottom_grad.scalar_type() == at::kHalf),
              "Data type of bottom_grad should be Float ro Half, got ",
              bottom_grad.scalar_type(), ".");

  TORCH_CHECK(bottom_grad.dim() == 4, "bottom_grad should be a 4d tensor, got ",
              top_grad.dim(), "D.");
  TORCH_CHECK(rois.dim() == 2, "rois should be a 2d tensor, got ", rois.dim(),
              "D.");
  TORCH_CHECK(top_grad.dim() == 4, "top_grad should be a 4d tensor, got ",
              bottom_grad.dim(), "D.");

  TORCH_CHECK((rois.size(0) == top_grad.size(0)),
              "the 1st dimensions of rois and top_grad should be the same, ",
              "but now the 1st dimension of rois is ", rois.size(0),
              ", and top_grad is ", top_grad.size(0), ".");

  TORCH_CHECK((bottom_grad.size(1) == top_grad.size(1)),
              "the 2nd dimensions of bottom_grad and top_grad should be ",
              "the same, but now the 2nd dimension of bottom_grad is ",
              bottom_grad.size(1), ", and top_grad is ", top_grad.size(1), ".");

  int channel = bottom_grad.size(1);
  int width = bottom_grad.size(3);
  int height = bottom_grad.size(2);
  int batch = bottom_grad.size(0);
  int rois_nums = rois.size(0);
  cnrtDataType_t d_type = torch_mlu::toCnrtDtype(bottom_grad.dtype());

  // return if zero-elements
  if (bottom_grad.numel() == 0) {
    CNLOG(INFO) << "Skip the zero-elements case.";
    return;
  }

  RoiAlignRotatedParams roiAlignRotatedParams{pooled_height,  pooled_width,
                                              sampling_ratio, spatial_scale,
                                              aligned,        clockwise};
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(rois_nums * pooled_height * pooled_width, &k_dim, &k_type);

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(top_grad.dim());
  auto top_grad_tensor =
      torch_mlu::cnnl::ops::cnnl_contiguous(top_grad, memory_format);
  at::Tensor bottom_grad_tmp = at::empty({batch, channel, height, width},
                                         top_grad.options(), memory_format)
                                   .zero_();

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto bottom_grad_impl = torch_mlu::getMluTensorImpl(bottom_grad_tmp);
  auto bottom_grad_ptr = bottom_grad_impl->cnnlMalloc();
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto rois_ptr = rois_impl->cnnlMalloc();
  auto top_grad_impl = torch_mlu::getMluTensorImpl(top_grad_tensor);
  auto top_grad_ptr = top_grad_impl->cnnlMalloc();

  KernelRoiAlignRotatedBackward(k_dim, k_type, queue, d_type, top_grad_ptr,
                                rois_ptr, bottom_grad_ptr, batch, height, width,
                                channel, rois_nums, roiAlignRotatedParams);
  bottom_grad.copy_(bottom_grad_tmp);
}

void roi_align_rotated_forward_mlu(Tensor input, Tensor rois, Tensor output,
                                   int aligned_height, int aligned_width,
                                   float spatial_scale, int sampling_ratio,
                                   bool aligned, bool clockwise) {
  ROIAlignRotatedForwardMLUKernelLauncher(input, rois, output, aligned_height,
                                          aligned_width, spatial_scale,
                                          sampling_ratio, aligned, clockwise);
}

void roi_align_rotated_backward_mlu(Tensor top_grad, Tensor rois,
                                    Tensor bottom_grad, int aligned_height,
                                    int aligned_width, float spatial_scale,
                                    int sampling_ratio, bool aligned,
                                    bool clockwise) {
  ROIAlignRotatedBackwardMLUKernelLauncher(
      top_grad, rois, bottom_grad, aligned_height, aligned_width, spatial_scale,
      sampling_ratio, aligned, clockwise);
}

void roi_align_rotated_forward_impl(Tensor input, Tensor rois, Tensor output,
                                    int aligned_height, int aligned_width,
                                    float spatial_scale, int sampling_ratio,
                                    bool aligned, bool clockwise);

void roi_align_rotated_backward_impl(Tensor top_grad, Tensor rois,
                                     Tensor bottom_grad, int aligned_height,
                                     int aligned_width, float spatial_scale,
                                     int sampling_ratio, bool aligned,
                                     bool clockwise);

REGISTER_DEVICE_IMPL(roi_align_rotated_forward_impl, MLU,
                     roi_align_rotated_forward_mlu);
REGISTER_DEVICE_IMPL(roi_align_rotated_backward_impl, MLU,
                     roi_align_rotated_backward_mlu);

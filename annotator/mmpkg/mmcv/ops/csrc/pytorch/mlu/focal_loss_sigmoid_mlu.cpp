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
#include <string>
#include <vector>

#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

void KernelFocalLossSigmoidForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                   cnrtQueue_t queue,
                                   const cnrtDataType_t d_type,
                                   const void *input, const void *target,
                                   const void *weight, const int32_t N,
                                   const int32_t C, const float alpha,
                                   const float gamma, void *output);

void KernelFocalLossSigmoidBackward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                    cnrtQueue_t queue,
                                    const cnrtDataType_t d_type,
                                    const void *input, const void *target,
                                    const void *weight, const float gamma,
                                    const float alpha, const int32_t dim_n,
                                    const int32_t deal_n, const int32_t dim_c,
                                    void *output);
// Policy Function for Forward
static void policyFuncForward(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                              const Tensor &input, const Tensor &target,
                              const Tensor &weight) {
  auto N = input.size(0);
  auto C = input.size(1);

  const size_t nram_size = torch_mlu::getDeviceAttr(cnrtAttrNramSizePerMcore);
  const size_t c_align_size = PAD_UP((C * input.itemsize()), NFU_ALIGN_SIZE);
  const int split_target_num = 2;
  const int split_pipeline_num = 6;
  const int has_weight = weight.data_ptr() != nullptr;
  const int target_data_width = target.scalar_type() == at::kLong
                                    ? target.itemsize() / 2
                                    : target.itemsize();
  const int threshold_c =
      PAD_DOWN((nram_size - split_target_num * sizeof(int)) /
                   (split_pipeline_num + has_weight),
               NFU_ALIGN_SIZE) /
      input.itemsize();

  int n_seg = 1;
  if (C <= threshold_c) {
    int c_size = C * input.itemsize();
    int reservered_align_size =
        (split_target_num + split_pipeline_num) * NFU_ALIGN_SIZE;
    int wegiht_size = 0;
    if (has_weight) {
      c_size = c_align_size;
      reservered_align_size = split_target_num * NFU_ALIGN_SIZE;
      wegiht_size = c_align_size;
    }
    // n_seg * c_size * split_pipeline_num + n_seg * target.itemsize() *
    // split_target_num
    //     + weight_size + reservered_align_size <= nram_size
    n_seg = (nram_size - wegiht_size - reservered_align_size) /
            (split_pipeline_num * c_size + split_target_num * sizeof(int32_t));
  }
  auto seg_num = n_seg == 0 ? N : (N + n_seg - 1) / n_seg;
  auto core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  auto core_num = core_dim * cluster_num;

  k_dim->x = *k_type;
  k_dim->y =
      seg_num > core_num ? cluster_num : (seg_num + core_dim - 1) / core_dim;
  k_dim->z = 1;
}

// Policy Function for Backward
static void policyFuncBackward(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  // set Union1 Job
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  k_dim->z = 1;
}

void SigmoidFocalLossForwardMLUKernelLauncher(Tensor input, Tensor target,
                                              Tensor weight, Tensor output,
                                              const float gamma,
                                              const float alpha) {
  // params check
  TORCH_CHECK(gamma >= 0, "gamma should be greater than or equal to 0. ",
              "But now gamma is ", gamma, ".");

  // check dtype
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "Data type of input should be Float or Half. But now input type is ",
      input.scalar_type(), ".");

  TORCH_CHECK(
      (target.scalar_type() == at::kInt || target.scalar_type() == at::kLong),
      "target type should be Int or Long. ", "But now target type is ",
      target.scalar_type(), ".");

  if (weight.data_ptr() != nullptr) {
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "Data types of input and weight should be the same. But now "
                "input type is ",
                input.scalar_type(), ", weight type is ", weight.scalar_type(),
                ".");
  } else {
    CNLOG(INFO) << "weight is a empty tensor.";
  }

  // return if zero-element
  if (input.numel() == 0 || target.numel() == 0 || output.numel() == 0) {
    return;
  }

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  policyFuncForward(&k_dim, &k_type, input, target, weight);
  auto core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input);
  auto input_ptr = input_impl->cnnlMalloc();
  auto target_impl = torch_mlu::getMluTensorImpl(target);
  auto target_ptr = target_impl->cnnlMalloc();
  auto weight_impl = torch_mlu::getMluTensorImpl(weight);
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();

  // get dtype of input
  cnrtDataType_t d_type = torch_mlu::toCnrtDtype(input.dtype());

  CNLOG(INFO) << "Launch Kernel KernelFocalLossSigmoidForward<<<Union"
              << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
              << k_dim.z << ">>>";
  // launch kernel
  KernelFocalLossSigmoidForward(k_dim, k_type, queue, d_type, input_ptr,
                                target_ptr, weight_ptr, input.size(0),
                                input.size(1), alpha, gamma, output_ptr);
}

void getDealNAndThresholdC(const int compute_data_bytes,
                           const int target_data_bytes, const int total_c,
                           int *deal_n_ptr, int *threshold_c_ptr,
                           const bool has_weight, const bool is_half) {
  /* NRAM partition:
   *
   * |-----------------ping pong--------------------|
   * |input | pt | alpha_t | temp | output | target | flt_min | gamma | weight|
   *
   * split_pipeline_num is 5: including input, pt, alpha_t, temp, output.
   */
  const int nram_split_num = 5;
  const int nram_split_pingpong = 2;
  const int max_nram_size = torch_mlu::getDeviceAttr(cnrtAttrNramSizePerMcore);
  int32_t compute_align_size = NFU_ALIGN_SIZE;
  if (is_half) {
    compute_align_size += NFU_ALIGN_SIZE;
  }
  const int32_t compute_align_num = compute_align_size / compute_data_bytes;
  // reservered_align_size: including input(ping pong), pt(ping pong),
  //                        alpha_t(ping pong), temp(ping pong),
  //                        output(ping pong), target(ping pong),
  //                        flt_min and gamma.
  const int reservered_align_size =
      ((nram_split_num + 1) * nram_split_pingpong + 2) * compute_align_size;
  int nram_pingpong_size = max_nram_size - reservered_align_size;

  int compute_c = total_c;
  int threshold_c = 0;
  if (has_weight) {
    // reserved space for weight to align
    nram_pingpong_size -= NFU_ALIGN_SIZE;

    // threshold_c * nram_split_pingpong * compute_data_bytes * nram_split_num +
    //     nram_split_pingpong * target_data_bytes +
    //     threshold_c * compute_data_bytes <= nram_pingpong_size
    threshold_c =
        (nram_pingpong_size - nram_split_pingpong * target_data_bytes) /
        (compute_data_bytes * (nram_split_num * nram_split_pingpong + 1));
    threshold_c = PAD_DOWN(threshold_c, compute_align_num);
    int weight_space = PAD_UP(total_c * compute_data_bytes, NFU_ALIGN_SIZE);

    // reserved space for weight
    nram_pingpong_size -= weight_space;
    compute_c = PAD_UP(total_c, compute_align_num);
  } else {
    // threshold_c * nram_split_pingpong * compute_data_bytes * nram_split_num +
    //     nram_split_pingpong * target_data_bytes <= nram_pingpong_size
    threshold_c =
        (nram_pingpong_size / nram_split_pingpong - target_data_bytes) /
        (nram_split_num * compute_data_bytes);
  }
  // deal_n * compute_c * nram_split_pingpong * compute_data_bytes *
  //     nram_split_num + deal_n * nram_split_pingpong * target_data_bytes <=
  //     nram_pingpong_size
  *deal_n_ptr =
      nram_pingpong_size /
      ((nram_split_num * compute_c * compute_data_bytes + target_data_bytes) *
       nram_split_pingpong);
  *threshold_c_ptr = threshold_c;
}

void SigmoidFocalLossBackwardMLUKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  // params check
  TORCH_CHECK(gamma >= 0, "gamma should be greater than or equal to 0. ",
              "But now gamma is ", gamma, ".");
  // check dtype
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "Data type of input should be Float or Half. But now input type is ",
      input.scalar_type(), ".");

  TORCH_CHECK(
      (target.scalar_type() == at::kInt || target.scalar_type() == at::kLong),
      "target type should be Int or Long. ", "But now target type is ",
      target.scalar_type(), ".");

  bool has_weight = false;
  if (weight.data_ptr() != nullptr) {
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "Data types of input and weight should be the same. But now "
                "input type is ",
                input.scalar_type(), ", weight type is ", weight.scalar_type(),
                ".");
    has_weight = true;
  } else {
    CNLOG(INFO) << "weight is a empty tensor.";
  }

  auto dim_c = input.size(1);
  const int compute_data_bytes = sizeof(float);
  // target supports only INT on MLU device while it keeps LONG on host side,
  // so target.itemsize() / 2
  const int target_data_bytes = target.scalar_type() == at::kLong
                                    ? (target.itemsize() / 2)
                                    : target.itemsize();
  int deal_n = 0;
  int threshold_c = 0;
  bool is_half = false;
  if (input.scalar_type() == at::kHalf) {
    is_half = true;
  }
  // calculate deal_n and threshold_c
  getDealNAndThresholdC(compute_data_bytes, target_data_bytes, dim_c, &deal_n,
                        &threshold_c, has_weight, is_half);

  // check C
  TORCH_CHECK(threshold_c >= dim_c,
              "input.size(1) should be in the range of [0, ", threshold_c,
              "]. ", "But now input.size(1) is ", dim_c, ".");

  if (input.numel() == 0 || target.numel() == 0 || output.numel() == 0) {
    // return if zero-element
    return;
  }

  // set task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncBackward(&k_dim, &k_type);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input);
  auto input_ptr = input_impl->cnnlMalloc();
  auto target_impl = torch_mlu::getMluTensorImpl(target);
  auto target_ptr = target_impl->cnnlMalloc();
  auto weight_impl = torch_mlu::getMluTensorImpl(weight);
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();

  // get dtype of input
  cnrtDataType_t d_type = torch_mlu::toCnrtDtype(input.dtype());
  auto core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto dim_n = input.size(0);

  CNLOG(INFO) << "Launch Kernel KernelFocalLossSigmoidBackward<<<Union"
              << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
              << k_dim.z << ">>>";

  // launch kernel
  KernelFocalLossSigmoidBackward(k_dim, k_type, queue, d_type, input_ptr,
                                 target_ptr, weight_ptr, gamma, alpha, dim_n,
                                 deal_n, dim_c, output_ptr);
}

void sigmoid_focal_loss_forward_mlu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, float gamma, float alpha) {
  SigmoidFocalLossForwardMLUKernelLauncher(input, target, weight, output, gamma,
                                           alpha);
}

void sigmoid_focal_loss_backward_mlu(Tensor input, Tensor target, Tensor weight,
                                     Tensor grad_input, float gamma,
                                     float alpha) {
  SigmoidFocalLossBackwardMLUKernelLauncher(input, target, weight, grad_input,
                                            gamma, alpha);
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, MLU,
                     sigmoid_focal_loss_forward_mlu);
REGISTER_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, MLU,
                     sigmoid_focal_loss_backward_mlu);

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
#include "carafe_utils.hpp"
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

void KernelCarafeForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                         cnrtQueue_t queue, const cnrtDataType_t d_type,
                         const void *input, const void *mask,
                         const CarafeForwardParam &param,
                         const CarafeForwardBlockDim &block_dim,
                         const CarafeForwardGridDim &grid_dim, void *output);

void KernelCarafeBackward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                          cnrtQueue_t queue, cnrtDataType_t dtype,
                          const void *input, const void *mask,
                          const void *grad_output, void *grad_input,
                          void *grad_mask, const int n, const int hi,
                          const int wi, const int c, const int k_up,
                          const int group, const int scale);

// Get total NRAM usage and set strides of NRAM arrays.
static void getNramUsage(CarafeForwardParam *param,
                         CarafeForwardBlockDim *block_dim, int *nram_usage) {
  // input_nram[blkDim_(Hi+Kh)-1, blkDim_(Wi+Kw)-1, blkDim_G, blkDim_Cg]
  block_dim->Hi = CEIL_DIV(block_dim->Ho, param->scale_factor) + 1;
  block_dim->Wi = CEIL_DIV(block_dim->Wo, param->scale_factor) + 1;

  param->input_nram_stride_g = PAD_UP(block_dim->Cg, param->align_size_NRAM);
  param->input_nram_stride_w = param->input_nram_stride_g * block_dim->G;
  param->input_nram_stride_h =
      (block_dim->Wi + block_dim->Kw - 1) * param->input_nram_stride_w;
  param->input_nram_size =
      (block_dim->Hi + block_dim->Kh - 1) * param->input_nram_stride_h;

  // mask_nram[blkDim_Ho, blkDim_Wo, blkDim_G, blkDim_Kh, blkDim_Kw]
  param->mask_nram_stride_kh = block_dim->Kw;
  param->mask_nram_stride_g = block_dim->Kh * param->mask_nram_stride_kh;
  param->mask_nram_stride_w = block_dim->G * param->mask_nram_stride_g;
  param->mask_nram_stride_h = block_dim->Wo * param->mask_nram_stride_w;
  param->mask_nram_size =
      PAD_UP(block_dim->Ho * param->mask_nram_stride_h, param->align_size_NRAM);

  // output_nram[blkDim_Ho, blkDim_Wo, blkDim_(G*Cg)]
  param->output_nram_stride_g = param->input_nram_stride_g;
  param->output_nram_stride_w =
      PAD_UP(param->input_nram_stride_w, param->align_size_NFU);
  param->output_nram_stride_h = block_dim->Wo * param->output_nram_stride_w;
  param->output_nram_size = block_dim->Ho * param->output_nram_stride_h;

  // sum_array[blkDim_(G*Cg)]

  // ensure the last mul_const on Cg does not exceed memory boundary
  int sum_array_size_bang_mul_const =
      (block_dim->G - 1) * param->input_nram_stride_g +
      PAD_UP(param->input_nram_stride_g, param->align_size_NFU);

  int sum_array_size =
      std::max(param->output_nram_stride_w, sum_array_size_bang_mul_const);

  *nram_usage = param->input_nram_size + param->mask_nram_size +
                param->output_nram_size + sum_array_size;
}

// Policy Function for Forward
static void genPolicyForward(CarafeForwardParam *param,
                             CarafeForwardBlockDim *block_dim,
                             CarafeForwardGridDim *grid_dim, cnrtDim3_t *k_dim,
                             cnrtFunctionType_t *k_type) {
  // device info
  auto core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  auto core_num = core_dim * cluster_num;

  // maximum NRAM size as the number of <dtype>
  auto max_nram_size =
      torch_mlu::getDeviceAttr(cnrtAttrNramSizePerMcore) / param->dtype_size;

  // determine grid and block dimensions

  // set initial values for block_dim and grid_dim
  block_dim->Ho = param->Ho;
  block_dim->Wo = param->Wo;
  block_dim->Kh = param->kernel_size;
  block_dim->Kw = param->kernel_size;
  block_dim->G = param->group_size;
  block_dim->Cg = param->Cg;

  grid_dim->Ho = 1;
  grid_dim->Wo = 1;
  grid_dim->Kh = 1;
  grid_dim->Kw = 1;
  grid_dim->G = 1;
  grid_dim->Cg = 1;

  // decrease the block size to fit in the NRAM.
  int nram_usage = 0;
  while (true) {
    getNramUsage(param, block_dim, &nram_usage);

    if (nram_usage > max_nram_size) {
      // decrease Ho
      // decrease block_Ho and block_Wo evenly
      // so that the block is close to a square.
      if (block_dim->Ho > 1 && block_dim->Ho >= block_dim->Wo) {
        grid_dim->Ho += 1;
        block_dim->Ho = CEIL_DIV(param->Ho, grid_dim->Ho);
      } else if (block_dim->Wo > 1 && block_dim->Wo > block_dim->Ho) {
        // decrease Wo
        grid_dim->Wo += 1;
        block_dim->Wo = CEIL_DIV(param->Wo, grid_dim->Wo);
      } else if (block_dim->Kh > 1) {
        // decrease Kh
        grid_dim->Kh += 1;
        block_dim->Kh = CEIL_DIV(param->kernel_size, grid_dim->Kh);
        // reset Hi, Wi to maximize NRAM usage
        grid_dim->Ho = 1;
        block_dim->Ho = param->Ho;
        grid_dim->Wo = 1;
        block_dim->Wo = param->Wo;
      } else if (block_dim->Kw > 1) {
        // decrease Kw
        grid_dim->Kw += 1;
        block_dim->Kw = CEIL_DIV(param->kernel_size, grid_dim->Kw);
        // reset Kh
        grid_dim->Kh = 1;
        block_dim->Kh = param->kernel_size;
      } else if (block_dim->G > 1) {
        // decrease G
        grid_dim->G += 1;
        block_dim->G = CEIL_DIV(param->group_size, grid_dim->G);
        // reset Kw
        grid_dim->Kw = 1;
        block_dim->Kw = param->kernel_size;
      } else if (block_dim->Cg > 1) {
        // decrease block_Cg
        // This is done in the last since c is the continuous dim
        // (input layout is NHWC) and large c can improve
        // IO & compute efficiency.
        grid_dim->Cg += 1;
        block_dim->Cg = CEIL_DIV(param->Cg, grid_dim->Cg);
        // reset G
        grid_dim->G = 1;
        block_dim->G = param->group_size;
      } else {
        // the block volume is one now, cannot decrease the block size anymore!
        // this situation should not occur.
        break;
      }
    } else {
      break;
    }
  }

  // define parameters depending on block_dim, grid_dim
  param->block_Cg_NFU = PAD_UP(block_dim->Cg, param->align_size_NFU);

  // define host arrays' strides

  // input[N,H,W,G,Cg]
  param->input_stride_g = param->Cg;
  param->input_stride_w = param->Ci;
  param->input_stride_h = param->Wi * param->input_stride_w;
  param->input_stride_n = param->Hi * param->input_stride_h;
  // mask[N,Ho,Wo,G,Kh,Kw]
  param->mask_stride_kh = param->kernel_size;
  param->mask_stride_g = param->kernel_size * param->mask_stride_kh;
  param->mask_stride_w = param->group_size * param->mask_stride_g;
  param->mask_stride_h = param->Wo * param->mask_stride_w;
  param->mask_stride_n = param->Ho * param->mask_stride_h;
  // output[N,Ho,Wo,G,Cg]
  param->output_stride_g = param->Cg;
  param->output_stride_w = param->Ci;
  param->output_stride_h = param->Wo * param->output_stride_w;
  param->output_stride_n = param->Ho * param->output_stride_h;

  param->job_num =
      param->N * grid_dim->Ho * grid_dim->Wo * grid_dim->G * grid_dim->Cg;

  // determine task type and dims
  *k_type = CNRT_FUNC_TYPE_BLOCK;
  k_dim->x = std::min(param->job_num, static_cast<int>(core_num));
  k_dim->y = 1;
  k_dim->z = 1;
}

void CARAFEForwardMLUKernelLauncher(const Tensor input, const Tensor mask,
                                    Tensor rinput, Tensor routput, Tensor rmask,
                                    Tensor output, const int kernel_size,
                                    const int group_size,
                                    const int scale_factor) {
  const int batch_size = output.size(0);
  const int channels = output.size(1);
  const int ho = output.size(2);
  const int wo = output.size(3);

  // check tensor data type
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "Data type of input should be Float or Half. But now input type is ",
      input.scalar_type(), ".");

  TORCH_CHECK(mask.scalar_type() == input.scalar_type(),
              "Data types of input and mask should be the same, but got ",
              input.scalar_type(), " and ", mask.scalar_type());

  // check number of dimensions
  TORCH_CHECK(input.dim() == 4, "input should be a 4-D tensor, but has ",
              input.dim(), "D.");
  TORCH_CHECK(mask.dim() == 4, "mask should be a 4-D tensor, but has ",
              input.dim(), "D.");

  // return fast on zero-element tensor
  if (output.numel() == 0) {
    output = at::zeros({batch_size, channels, ho, wo}, output.options());
    return;
  }

  // set param
  CarafeForwardParam param;
  param.N = input.size(0);
  param.Ci = input.size(1);
  param.Hi = input.size(2);
  param.Wi = input.size(3);

  param.kernel_size = kernel_size;
  param.group_size = group_size;
  param.scale_factor = scale_factor;
  param.Cg = param.Ci / group_size;
  param.dtype_size = input.itemsize();
  param.align_size_NRAM = NRAM_ALIGN_SIZE / param.dtype_size;
  param.align_size_NFU = NFU_ALIGN_SIZE / param.dtype_size;
  param.kernel_size_sq = param.kernel_size * param.kernel_size;
  param.kernel_size_half = (param.kernel_size - 1) / 2;
  param.Ho = param.Hi * param.scale_factor;
  param.Wo = param.Wi * param.scale_factor;

  // generate policy
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  CarafeForwardBlockDim block_dim;
  CarafeForwardGridDim grid_dim;

  genPolicyForward(&param, &block_dim, &grid_dim, &k_dim, &k_type);

  // convert NCHW to NHWC
  auto memory_format_input_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto rinput_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format_input_nhwc);

  auto memory_format_mask_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(mask.dim());
  auto rmask_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(mask, memory_format_mask_nhwc);

  auto memory_format_output_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(output.dim());
  auto routput_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(output, memory_format_output_nhwc);

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(rinput_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto mask_impl = torch_mlu::getMluTensorImpl(rmask_);
  auto mask_ptr = mask_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(routput_);
  auto output_ptr = output_impl->cnnlMalloc();

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get dtype of input
  cnrtDataType_t d_type = torch_mlu::toCnrtDtype(input.dtype());

  // launch kernel
  auto core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  CNLOG(INFO) << "Launch Kernel KernelCarafeForward<<<Union"
              << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
              << k_dim.z << ">>>";

  KernelCarafeForward(k_dim, k_type, queue, d_type, input_ptr, mask_ptr, param,
                      block_dim, grid_dim, output_ptr);

  // copy output from NHWC back into NCHW
  rinput.copy_(rinput_);
  output.copy_(routput_);
}

// Policy Function for Backward
static void policyFuncBackward(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  // set Union1 Job
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  k_dim->z = 1;
}

void CARAFEBackwardMLUKernelLauncher(
    const Tensor grad_output, const Tensor rinput, const Tensor mask,
    Tensor rgrad_output, Tensor rgrad_input_hs, Tensor rgrad_input,
    Tensor rgrad_mask, Tensor grad_input, Tensor grad_mask,
    const int kernel_size, const int group_size, const int scale_factor) {
  const int batch_size = rinput.size(0);
  const int channels = rinput.size(1);
  const int hi = rinput.size(2);
  const int wi = rinput.size(3);

  // data type check
  TORCH_CHECK(grad_output.scalar_type() == at::kFloat ||
                  grad_output.scalar_type() == at::kHalf,
              "grad_output type should be Float or Half, got ",
              grad_output.scalar_type());
  TORCH_CHECK(grad_output.scalar_type() == mask.scalar_type(),
              "mask should have the same type as grad_output");

  // dim check
  TORCH_CHECK(grad_output.dim() == 4, "grad_output should be a 4d tensor, got ",
              grad_output.dim(), "D");

  // param check
  TORCH_CHECK(kernel_size < 137, "kernel_size should be less than 137, got ",
              kernel_size);

  // set task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncBackward(&k_dim, &k_type);

  // convert NCHW to NHWC
  auto memory_format_input_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(rinput.dim());
  auto rinput_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(rinput, memory_format_input_nhwc);

  auto memory_format_mask_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(mask.dim());
  auto rmask_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(mask, memory_format_mask_nhwc);

  auto memory_format_grad_output_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_output.dim());
  auto rgrad_output_ = torch_mlu::cnnl::ops::cnnl_contiguous(
      grad_output, memory_format_grad_output_nhwc);

  auto memory_format_grad_input_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_input.dim());
  auto rgrad_input_ = torch_mlu::cnnl::ops::cnnl_contiguous(
                          grad_input, memory_format_grad_input_nhwc)
                          .zero_();

  auto memory_format_grad_mask_nhwc =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_mask.dim());
  auto rgrad_mask_ = torch_mlu::cnnl::ops::cnnl_contiguous(
      grad_mask, memory_format_grad_mask_nhwc);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(rinput_);
  auto input_ptr = input_impl->cnnlMalloc();
  auto mask_impl = torch_mlu::getMluTensorImpl(rmask_);
  auto mask_ptr = mask_impl->cnnlMalloc();
  auto grad_output_impl = torch_mlu::getMluTensorImpl(rgrad_output_);
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto grad_input_impl = torch_mlu::getMluTensorImpl(rgrad_input_);
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  auto grad_mask_impl = torch_mlu::getMluTensorImpl(rgrad_mask_);
  auto grad_mask_ptr = grad_mask_impl->cnnlMalloc();

  // get dtype of grad_output
  cnrtDataType_t d_type = torch_mlu::toCnrtDtype(grad_output.dtype());
  auto core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);

  CNLOG(INFO) << "Launch Kernel KernelCarafeBackward<<<Union"
              << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
              << k_dim.z << ">>>";

  // launch kernel
  KernelCarafeBackward(k_dim, k_type, queue, d_type, input_ptr, mask_ptr,
                       grad_output_ptr, grad_input_ptr, grad_mask_ptr,
                       batch_size, hi, wi, channels, kernel_size, group_size,
                       scale_factor);

  // copy output from NHWC back into NCHW
  grad_input.copy_(rgrad_input_);
  grad_mask.copy_(rgrad_mask_);
}

void carafe_forward_mlu(Tensor features, Tensor masks, Tensor rfeatures,
                        Tensor routput, Tensor rmasks, Tensor output,
                        int kernel_size, int group_size, int scale_factor) {
  CARAFEForwardMLUKernelLauncher(features, masks, rfeatures, routput, rmasks,
                                 output, kernel_size, group_size, scale_factor);
}

void carafe_backward_mlu(Tensor top_grad, Tensor rfeatures, Tensor masks,
                         Tensor rtop_grad, Tensor rbottom_grad_hs,
                         Tensor rbottom_grad, Tensor rmask_grad,
                         Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                         int group_size, int scale_factor) {
  CARAFEBackwardMLUKernelLauncher(top_grad, rfeatures, masks, rtop_grad,
                                  rbottom_grad_hs, rbottom_grad, rmask_grad,
                                  bottom_grad, mask_grad, kernel_size,
                                  group_size, scale_factor);
}

void carafe_forward_impl(Tensor features, Tensor masks, Tensor rfeatures,
                         Tensor routput, Tensor rmasks, Tensor output,
                         int kernel_size, int group_size, int scale_factor);

void carafe_backward_impl(Tensor top_grad, Tensor rfeatures, Tensor masks,
                          Tensor rtop_grad, Tensor rbottom_grad_hs,
                          Tensor rbottom_grad, Tensor rmask_grad,
                          Tensor bottom_grad, Tensor mask_grad, int kernel_size,
                          int group_size, int scale_factor);

REGISTER_DEVICE_IMPL(carafe_forward_impl, MLU, carafe_forward_mlu);
REGISTER_DEVICE_IMPL(carafe_backward_impl, MLU, carafe_backward_mlu);

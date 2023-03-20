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

void KernelMaskedIm2colForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    cnrtDataType_t k_dtype, const void *im_ptr, const int height,
    const int width, const int channels, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const void *mask_h_idx_ptr,
    const void *mask_w_idx_ptr, const int mask_cnt, void *col_ptr);

void KernelMaskedCol2imForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                               cnrtQueue_t queue, cnrtDataType_t k_dtype,
                               const void *col_ptr, const int height,
                               const int width, const int channels,
                               const void *mask_h_idx_ptr,
                               const void *mask_w_idx_ptr, const int mask_cnt,
                               void *im_ptr);

// policy function
static void policyFunc(const int mask_cnt, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  const size_t cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  const size_t core_num = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  const size_t task_dim = CEIL_ALIGN(mask_cnt, core_num);
  k_dim->x = core_num;
  k_dim->y =
      (task_dim / core_num) > cluster_num ? cluster_num : (task_dim / core_num);
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

void MaskedIm2colForwardMLUKernelLauncher(const Tensor im,
                                          const Tensor mask_h_idx,
                                          const Tensor mask_w_idx, Tensor col,
                                          const int kernel_h,
                                          const int kernel_w, const int pad_h,
                                          const int pad_w) {
  // Check dtype.
  TORCH_CHECK(im.scalar_type() == at::kFloat || im.scalar_type() == at::kHalf,
              "im type should be Float or Half, got ", im.scalar_type(), ".");
  TORCH_CHECK(mask_h_idx.scalar_type() == at::kInt ||
                  mask_h_idx.scalar_type() == at::kLong,
              "mask_h_idx type should be Int or Long, got ",
              mask_h_idx.scalar_type(), ".");
  TORCH_CHECK(mask_w_idx.scalar_type() == at::kInt ||
                  mask_w_idx.scalar_type() == at::kLong,
              "mask_w_idx type should be Int or Long, got ",
              mask_w_idx.scalar_type(), ".");
  TORCH_CHECK(kernel_h > 0, "kernel_h should greater than 0, got ", kernel_h,
              ".");
  TORCH_CHECK(kernel_w > 0, "kernel_w should greater than 0, got ", kernel_w,
              ".");

  // zero element check
  TORCH_CHECK(im.numel() > 0, "im.numel should greater than zero, got ",
              im.numel(), ".");
  TORCH_CHECK(col.size(0) > 0, "col.size(0) should greater than zero, got ",
              col.size(0), ".");

  // large tensor check
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  TORCH_CHECK(im.numel() < max_input_num,
              "im.numel() should be less than 2147483648, got ", im.numel(),
              ".");
  TORCH_CHECK(col.numel() < max_input_num,
              "col.numel() should be less than 2147483648, got ", col.numel(),
              ".");

  const int channels = im.size(1);
  const int height = im.size(2);
  const int width = im.size(3);
  const int mask_cnt = mask_h_idx.size(0);

  // auto im_t = im.permute({0, 2, 3, 1}).contiguous();
  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(im.dim());
  auto im_ = torch_mlu::cnnl::ops::cnnl_contiguous(im, memory_format);
  auto col_ =
      at::zeros({mask_cnt, kernel_h * kernel_w, channels}, col.options());
  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(mask_cnt, &k_dim, &k_type);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();
  // get ptr of tensors
  auto im_impl = torch_mlu::getMluTensorImpl(im_);
  auto im_ptr = im_impl->cnnlMalloc();
  auto mask_h_idx_impl = torch_mlu::getMluTensorImpl(mask_h_idx);
  auto mask_h_idx_ptr = mask_h_idx_impl->cnnlMalloc();
  auto mask_w_idx_impl = torch_mlu::getMluTensorImpl(mask_w_idx);
  auto mask_w_idx_ptr = mask_w_idx_impl->cnnlMalloc();
  auto col_impl = torch_mlu::getMluTensorImpl(col_);
  auto col_ptr = col_impl->cnnlMalloc();

  // get comput dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(im.dtype());

  // launch kernel
  CNLOG(INFO) << "Launch Kernel MLUKernelMaskedIm2colForward<<<" << k_dim.x
              << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  KernelMaskedIm2colForward(k_dim, k_type, queue, data_type, im_ptr, height,
                            width, channels, kernel_h, kernel_w, pad_h, pad_w,
                            mask_h_idx_ptr, mask_w_idx_ptr, mask_cnt, col_ptr);

  col.copy_(col_.permute({2, 1, 0})
                .reshape({channels * kernel_h * kernel_w, mask_cnt})
                .contiguous());
}

void MaskedCol2imForwardMLUKernelLauncher(const Tensor col,
                                          const Tensor mask_h_idx,
                                          const Tensor mask_w_idx, Tensor im,
                                          const int height, const int width,
                                          const int channels) {
  // Check dtype.
  TORCH_CHECK(col.scalar_type() == at::kFloat || col.scalar_type() == at::kHalf,
              "col type should be Float or Half, got ", col.scalar_type(), ".");
  TORCH_CHECK(mask_h_idx.scalar_type() == at::kInt ||
                  mask_h_idx.scalar_type() == at::kLong,
              "mask_h_idx type should be Int or Long, got ",
              mask_h_idx.scalar_type(), ".");
  TORCH_CHECK(mask_w_idx.scalar_type() == at::kInt ||
                  mask_w_idx.scalar_type() == at::kLong,
              "mask_w_idx type should be Int or Long, got ",
              mask_w_idx.scalar_type(), ".");

  // zero element check
  TORCH_CHECK(im.numel() > 0, "im.numel should greater than zero, got ",
              im.numel(), ".");
  TORCH_CHECK(col.size(0) > 0, "col.size(0) should greater than zero, got ",
              col.size(0), ".");

  // large tensor check
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  TORCH_CHECK(im.numel() < max_input_num,
              "im.numel() should be less than 2147483648, got ", im.numel(),
              ".");
  TORCH_CHECK(col.numel() < max_input_num,
              "col.numel() should be less than 2147483648, got ", col.numel(),
              ".");

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(im.dim());
  at::Tensor im_ =
      at::empty({1, channels, height, width}, im.options(), memory_format)
          .zero_();

  auto col_t = torch_mlu::cnnl::ops::cnnl_contiguous(col.transpose(0, 1));

  const int mask_cnt = mask_h_idx.size(0);
  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(mask_cnt, &k_dim, &k_type);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();
  // get ptr of tensors
  auto im_impl = torch_mlu::getMluTensorImpl(im_);
  auto im_ptr = im_impl->cnnlMalloc();
  auto mask_h_idx_impl = torch_mlu::getMluTensorImpl(mask_h_idx);
  auto mask_h_idx_ptr = mask_h_idx_impl->cnnlMalloc();
  auto mask_w_idx_impl = torch_mlu::getMluTensorImpl(mask_w_idx);
  auto mask_w_idx_ptr = mask_w_idx_impl->cnnlMalloc();
  auto col_t_impl = torch_mlu::getMluTensorImpl(col_t);
  auto col_t_ptr = col_t_impl->cnnlMalloc();

  // get comput dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(col.dtype());

  // launch kernel
  CNLOG(INFO) << "Launch Kernel MLUKernelMaskedCol2imForward<<<" << k_dim.x
              << ", " << k_dim.y << ", " << k_dim.z << ">>>";

  KernelMaskedCol2imForward(k_dim, k_type, queue, data_type, col_t_ptr, height,
                            width, channels, mask_h_idx_ptr, mask_w_idx_ptr,
                            mask_cnt, im_ptr);

  im.copy_(im_);
}

void masked_im2col_forward_mlu(const Tensor im, const Tensor mask_h_idx,
                               const Tensor mask_w_idx, Tensor col,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)
  MaskedIm2colForwardMLUKernelLauncher(im, mask_h_idx, mask_w_idx, col,
                                       kernel_h, kernel_w, pad_h, pad_w);
}

void masked_col2im_forward_mlu(const Tensor col, const Tensor mask_h_idx,
                               const Tensor mask_w_idx, Tensor im, int height,
                               int width, int channels) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)
  MaskedCol2imForwardMLUKernelLauncher(col, mask_h_idx, mask_w_idx, im, height,
                                       width, channels);
}

void masked_im2col_forward_impl(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w);

void masked_col2im_forward_impl(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels);

REGISTER_DEVICE_IMPL(masked_im2col_forward_impl, MLU,
                     masked_im2col_forward_mlu);
REGISTER_DEVICE_IMPL(masked_col2im_forward_impl, MLU,
                     masked_col2im_forward_mlu);

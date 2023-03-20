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

void KernelThreeNNForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                          cnrtQueue_t queue, cnrtDataType_t data_type,
                          const void *unknown, const void *known, void *dist2,
                          int *idx, const int b, const int n, const int m);

void ThreeNNMLUKernelLauncher(int b, int n, int m, const Tensor unknown,
                              const Tensor known, Tensor dist2, Tensor idx) {
  // Check dtype.
  TORCH_CHECK(
      unknown.scalar_type() == at::kFloat || unknown.scalar_type() == at::kHalf,
      "unknown type should be Float or Half, got ", unknown.scalar_type(), ".");
  TORCH_CHECK(unknown.scalar_type() == known.scalar_type(),
              "known should have the same type as unknown.");
  TORCH_CHECK(unknown.scalar_type() == dist2.scalar_type(),
              "dist2 should have the same type as unknown.");
  TORCH_CHECK(idx.scalar_type() == at::kInt, "idx type should be Int.");

  // Check shape.
  TORCH_CHECK(unknown.dim() == 3, "unknown should be 3d tensor, got ",
              unknown.dim(), "D.");
  TORCH_CHECK(known.dim() == 3, "known should be 3d tensor, got ", known.dim(),
              "D.");
  TORCH_CHECK(unknown.size(0) == known.size(0),
              "known.dim0 should be equal to unknown.dim0, got ", known.size(0),
              ".");
  TORCH_CHECK(unknown.size(2) == 3, "unknown dim2 should be 3, got ",
              unknown.size(2), ".");
  TORCH_CHECK(known.size(2) == 3, "known dim2 should be 3, got ", known.size(2),
              ".");

  // zero element check
  TORCH_CHECK(unknown.numel() > 0,
              "unknown.numel should greater than zero, got ", unknown.numel(),
              ".");
  if (known.numel() == 0) {
    // return if known zero element
    return;
  }

  // large tensor check
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  TORCH_CHECK(unknown.numel() < max_input_num,
              "unknown.numel() should be less than 2147483648, got ",
              unknown.numel(), ".");
  TORCH_CHECK(known.numel() < max_input_num,
              "known.numel() should be less than 2147483648, got ",
              known.numel(), ".");

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto unknown_impl = torch_mlu::getMluTensorImpl(unknown);
  auto unknown_ptr = unknown_impl->cnnlMalloc();
  auto known_t = known.permute({0, 2, 1}).contiguous();
  auto known_impl = torch_mlu::getMluTensorImpl(known_t);
  auto known_ptr = known_impl->cnnlMalloc();
  auto dist2_impl = torch_mlu::getMluTensorImpl(dist2);
  auto dist2_ptr = dist2_impl->cnnlMalloc();
  auto idx_impl = torch_mlu::getMluTensorImpl(idx);
  auto idx_ptr = idx_impl->cnnlMalloc();

  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  k_dim.x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.y = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  k_dim.z = 1;
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(unknown.dtype());

  // launch kernel
  CNLOG(INFO) << "Launch Kernel MLUKernelThreeNNForward<<<" << k_dim.x << ", "
              << k_dim.y << ", " << k_dim.z << ">>>.";

  KernelThreeNNForward(k_dim, k_type, queue, data_type, unknown_ptr, known_ptr,
                       dist2_ptr, (int *)idx_ptr, b, n, m);
}

void three_nn_forward_mlu(int b, int n, int m, const Tensor unknown,
                          const Tensor known, Tensor dist2, Tensor idx) {
  ThreeNNMLUKernelLauncher(b, n, m, unknown, known, dist2, idx);
}

void three_nn_forward_impl(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx);

REGISTER_DEVICE_IMPL(three_nn_forward_impl, MLU, three_nn_forward_mlu);

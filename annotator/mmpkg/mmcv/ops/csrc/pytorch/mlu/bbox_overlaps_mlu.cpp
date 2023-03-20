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

void KernelBBoxOverlaps(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                        cnrtQueue_t queue, const cnrtDataType_t d_type,
                        const void *bbox1, const void *bbox2, void *ious,
                        const int32_t num_bbox1, const int32_t num_bbox2,
                        const int32_t mode, const bool aligned,
                        const int32_t offset);

static void policyFunc(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                       const int32_t batch_num_all) {
  auto union_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  auto core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto core_num = union_num * core_dim;

  // Union1 policyFunc
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_dim;
  auto need_core_num = PAD_UP(batch_num_all, core_dim);
  k_dim->y =
      (need_core_num < core_num) ? (need_core_num / core_dim) : union_num;
  k_dim->z = 1;

  return;
}

void BBoxOverlapsMLUKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                   Tensor ious, const int32_t mode,
                                   const bool aligned, const int32_t offset) {
  // check dtype
  TORCH_CHECK(
      bboxes1.scalar_type() == at::kFloat || bboxes1.scalar_type() == at::kHalf,
      "Data type of input should be Float or Half. But now input type is ",
      bboxes1.scalar_type(), ".");
  TORCH_CHECK(bboxes1.scalar_type() == bboxes2.scalar_type(),
              "bboxes1's dtype should be the same with bboxes2's dtype.");

  // params check
  TORCH_CHECK(bboxes1.dim() == 2, "bboxes1 should be a 2d tensor, got ",
              bboxes1.dim(), "D");
  TORCH_CHECK(bboxes2.dim() == 2, "bboxes2 should be a 2d tensor, got ",
              bboxes2.dim(), "D");

  auto rows = bboxes1.size(0);
  auto cols = bboxes2.size(0);
  auto batch_num_all = rows;

  if (rows * cols == 0) {
    // return if zero element
    return;
  }

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(&k_dim, &k_type, batch_num_all);

  // get compute queue
  cnrtQueue_t queue = torch_mlu::getCurQueue();

  // get dtype of input
  cnrtDataType_t d_type = torch_mlu::toCnrtDtype(bboxes1.dtype());

  // get ptr of tensors
  auto bboxes1_impl = torch_mlu::getMluTensorImpl(bboxes1);
  auto bboxes1_ptr = bboxes1_impl->cnnlMalloc();
  auto bboxes2_impl = torch_mlu::getMluTensorImpl(bboxes2);
  auto bboxes2_ptr = bboxes2_impl->cnnlMalloc();
  auto ious_impl = torch_mlu::getMluTensorImpl(ious);
  auto ious_ptr = ious_impl->cnnlMalloc();

  // launch kernel
  CNLOG(INFO) << "Launch Kernel MLUUnion1BboxOverlapsKernel";
  CNLOG(INFO) << "kDim :[ " << k_dim.x << ", " << k_dim.y << ", " << k_dim.z
              << " ]";
  KernelBBoxOverlaps(k_dim, k_type, queue, d_type, bboxes1_ptr, bboxes2_ptr,
                     ious_ptr, rows, cols, mode, aligned, offset);
}

void bbox_overlaps_mlu(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                       const int mode, const bool aligned, const int offset) {
  BBoxOverlapsMLUKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset);
REGISTER_DEVICE_IMPL(bbox_overlaps_impl, MLU, bbox_overlaps_mlu);

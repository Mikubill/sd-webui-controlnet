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

void KernelNms(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
               const cnrtDataType_t data_type_input, const void *boxes_ptr,
               const void *scores_ptr, const int input_num_boxes,
               const int max_output_boxes, const float iou_threshold,
               const float offset, void *workspace_ptr, void *output_size_ptr,
               void *output_ptr);

int selectUnionType(uint32_t use_job, int box_num_per_core) {
  // the box_num_per_core should be at least 256, otherwise the real IO
  // bandwidth would be very low
  while (box_num_per_core < 256 && use_job >= 4) {
    box_num_per_core *= 2;
    use_job /= 2;
  }
  return use_job;
}

static cnnlStatus_t policyFunc(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                               int &core_num_per_class,
                               const int input_box_num) {
  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  uint32_t cluster_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t job_limit = getJobLimitCapability();
  uint32_t core_number = job_limit;

  int box_num_per_core = (input_box_num + core_number - 1) / core_number;
  int use_job = selectUnionType(job_limit, box_num_per_core);
  // initiate k_type as Union1
  k_dim->x = core_dim;
  k_dim->y = 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  switch (job_limit) {
    case CN_KERNEL_CLASS_BLOCK:
    case CN_KERNEL_CLASS_UNION:
    case CN_KERNEL_CLASS_UNION2:
    case CN_KERNEL_CLASS_UNION4:
    case CN_KERNEL_CLASS_UNION8:
    case CN_KERNEL_CLASS_UNION16: {
      if (use_job < 4) {
        k_dim->x = 1;
        *k_type = CNRT_FUNC_TYPE_BLOCK;
      } else if (use_job == 4) {
        k_dim->x = core_dim;
        *k_type = CNRT_FUNC_TYPE_UNION1;
      } else {
        k_dim->x = use_job;
        *k_type = (cnrtFunctionType_t)use_job;
      }
    }; break;
    default:
      LOG(WARNING) << "[cnnlNms_v2]: got unsupported job limit number."
                   << " Use default CN_KERNEL_CLASS_UNION1 with UNION1 task.";
  }
  return CNNL_STATUS_SUCCESS;
}

Tensor NMSMLUKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                            int offset) {
  // dimension parameters check
  TORCH_CHECK(boxes.dim() == 2, "boxes should be a 2d tensor, got ",
              boxes.dim(), "D");
  TORCH_CHECK(boxes.size(1) == 4,
              "boxes should have 4 elements in dimension 1, got ",
              boxes.size(1));
  TORCH_CHECK(scores.dim() == 1, "scores should be a 1d tensor, got ",
              scores.dim(), "D");

  // data type check
  TORCH_CHECK(boxes.scalar_type() == scores.scalar_type(),
              "boxes should have the same type as scores");
  TORCH_CHECK(
      boxes.scalar_type() == at::kFloat || boxes.scalar_type() == at::kHalf,
      "data type of boxes should be Float or Half, got ", boxes.scalar_type());

  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }

  int input_num_boxes = boxes.size(0);
  int max_output_boxes = boxes.size(0);

  cnrtDataType_t data_type_input = torch_mlu::toCnrtDtype(boxes.dtype());
  cnrtDim3_t k_dim;
  cnrtJobType_t k_type;

  int core_num_per_class;
  policyFunc(&k_dim, &k_type, core_num_per_class, input_num_boxes);

  // transpose boxes (n, 4) to (4, n) for better performance
  auto boxes_t = boxes.transpose(0, 1);
  auto boxes_ = torch_mlu::cnnl::ops::cnnl_contiguous(boxes_t);
  auto scores_ = torch_mlu::cnnl::ops::cnnl_contiguous(scores);
  auto output = at::empty({max_output_boxes}, boxes.options().dtype(at::kLong));
  auto output_size = at::empty({1}, scores.options().dtype(at::kInt));

  // workspace
  const int info_num = 5;  // x1, x2, y1, y2 and score
  size_t space_size = 0;
  if (boxes.scalar_type() == at::kHalf) {
    space_size = input_num_boxes * sizeof(int16_t) * info_num + sizeof(float);
  } else {
    space_size = input_num_boxes * sizeof(float) * info_num + sizeof(float);
  }
#if __BANG_ARCH__ > 370
  int cluster_num = getCoreNumOfJobLimitCapability() /
                    torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  space_size += cluster_number * sizeof(float) * 7;
#endif
  auto workspace = at::empty(space_size, boxes.options().dtype(at::kByte));

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  auto boxes_impl = torch_mlu::getMluTensorImpl(boxes_);
  auto boxes_ptr = boxes_impl->cnnlMalloc();
  auto scores_impl = torch_mlu::getMluTensorImpl(scores_);
  auto scores_ptr = scores_impl->cnnlMalloc();
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  auto output_size_impl = torch_mlu::getMluTensorImpl(output_size);
  auto output_size_ptr = output_size_impl->cnnlMalloc();

  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  CNLOG(INFO) << "Launch Kernel MLUUnionX NMS<<<Union" << k_type / core_dim
              << ", " << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  KernelNms(k_dim, k_type, queue, data_type_input, boxes_ptr, scores_ptr,
            input_num_boxes, max_output_boxes, iou_threshold, offset,
            workspace_ptr, output_size_ptr, output_ptr);
  int output_num = *static_cast<int *>(output_size.cpu().data_ptr());
  return output.slice(0, 0, output_num);
}

Tensor nms_mlu(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return NMSMLUKernelLauncher(boxes, scores, iou_threshold, offset);
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_impl, MLU, nms_mlu);

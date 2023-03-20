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

void KernelIou3d(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
                 const cnrtDataType_t data_type_input, const void *boxes_dram,
                 const int input_box_num, const float iou_threshold,
                 void *workspace, void *output_size, void *output);

int selectType(uint32_t use_job, int box_num_per_core) {
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
  uint32_t job_limit = getJobLimitCapability();
  uint32_t core_number = job_limit;

  int box_num_per_core = (input_box_num + core_number - 1) / core_number;
  int use_job = selectType(job_limit, box_num_per_core);
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

void IoU3DNMS3DMLUKernelLauncher(Tensor boxes, Tensor &keep, Tensor &keep_num,
                                 float iou_threshold) {
  // dimension parameters check
  TORCH_CHECK(boxes.dim() == 2, "boxes should be a 2d tensor, got ",
              boxes.dim(), "D");
  TORCH_CHECK(boxes.size(1) == 7,
              "boxes should have 7 elements in dimension 1, got ",
              boxes.size(1));

  // data type check
  TORCH_CHECK(
      boxes.scalar_type() == at::kFloat || boxes.scalar_type() == at::kHalf,
      "data type of boxes should be Float or Half, got ", boxes.scalar_type());

  if (boxes.numel() == 0) {
    return;
  }
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  TORCH_CHECK(boxes.numel() < max_input_num,
              "boxes.numel() should be less than 2147483648, got ",
              boxes.numel());
  int input_box_num = boxes.size(0);

  cnrtDataType_t data_type_input = torch_mlu::toCnrtDtype(boxes.dtype());
  cnrtDim3_t k_dim;
  cnrtJobType_t k_type;

  int core_num_per_class;
  policyFunc(&k_dim, &k_type, core_num_per_class, input_box_num);

  // transpose boxes (n, 7) to (7, n) for better performance
  auto boxes_t = boxes.transpose(0, 1);
  auto boxes_ = torch_mlu::cnnl::ops::cnnl_contiguous(boxes_t);

  auto output = at::empty({input_box_num}, boxes.options().dtype(at::kLong));
  auto output_size = at::empty({1}, boxes.options().dtype(at::kInt));

  // workspace
  const int info_num = 7;  // x, y,z, dx, dy, dz,angle
  size_t space_size = 0;
  if (boxes.scalar_type() == at::kHalf) {
    space_size = input_box_num * sizeof(int16_t) * info_num +
                 input_box_num * sizeof(float) + sizeof(float);
  } else {
    space_size = input_box_num * sizeof(float) * (info_num + 1) + sizeof(float);
  }

  auto workspace = at::empty(space_size, boxes.options().dtype(at::kByte));

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  auto boxes_impl = torch_mlu::getMluTensorImpl(boxes_);
  auto boxes_ptr = boxes_impl->cnnlMalloc();
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(keep);
  auto output_ptr = output_impl->cnnlMalloc();
  auto output_size_impl = torch_mlu::getMluTensorImpl(keep_num);
  auto output_size_ptr = output_size_impl->cnnlMalloc();

  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  CNLOG(INFO) << "Launch Kernel KernelIou3d<<<Union" << k_type / core_dim
              << ", " << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  KernelIou3d(k_dim, k_type, queue, data_type_input, boxes_ptr, input_box_num,
              iou_threshold, workspace_ptr, output_size_ptr, output_ptr);
}

void iou3d_nms3d_forward_mlu(const Tensor boxes, Tensor &keep, Tensor &keep_num,
                             float nms_overlap_thresh) {
  IoU3DNMS3DMLUKernelLauncher(boxes, keep, keep_num, nms_overlap_thresh);
}

void iou3d_nms3d_forward_impl(const Tensor boxes, Tensor &keep,
                              Tensor &keep_num, float nms_overlap_thresh);
REGISTER_DEVICE_IMPL(iou3d_nms3d_forward_impl, MLU, iou3d_nms3d_forward_mlu);

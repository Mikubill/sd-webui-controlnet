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
#include <algorithm>

#include "psamask_utils.hpp"
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

#define COMPUTE_COUNT_ALIGN 64

void KernelPsamaskForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, const PsamaskType psa_type,
    const DimPartitionType core_partition,
    const DimPartitionType cluster_partition, const int batch,
    const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int x_c, const int y_c, const int half_h_mask,
    const int half_w_mask, const int n_per_core, const int h_per_core,
    const int n_per_cluster, const int h_per_cluster, const int limit_n_seg,
    const int limit_h_seg, const int limit_w_seg);

void KernelPsamaskBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *dy, void *dx, const PsamaskType psa_type,
    const DimPartitionType core_partition,
    const DimPartitionType cluster_partition, const int batch,
    const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int dx_c, const int dy_c, const int half_h_mask,
    const int half_w_mask, const int n_per_core, const int h_per_core,
    const int n_per_cluster, const int h_per_cluster, const int limit_n_seg,
    const int limit_h_seg, const int limit_w_seg);

namespace {
void policyFunc(cnrtDim3_t *k_dim_ptr, cnrtFunctionType_t *f_type_ptr,
                PartitionSeg *partition_ptr, const int n, const int h_feature) {
  unsigned int core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  unsigned int cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  unsigned int use_cluster_num = cluster_num;
  unsigned int use_core_num = core_dim;

  if (n >= cluster_num || n >= h_feature) {
    partition_ptr->cluster_partition = PARTITION_N;
    partition_ptr->n_per_cluster = (n + cluster_num - 1) / cluster_num;
    partition_ptr->h_per_cluster = h_feature;
    use_cluster_num =
        (n + partition_ptr->n_per_cluster - 1) / partition_ptr->n_per_cluster;
  } else {
    partition_ptr->cluster_partition = PARTITION_H;
    partition_ptr->h_per_cluster = (h_feature + cluster_num - 1) / cluster_num;
    partition_ptr->n_per_cluster = n;
    use_cluster_num = (h_feature + partition_ptr->h_per_cluster - 1) /
                      partition_ptr->h_per_cluster;
  }

  if (partition_ptr->n_per_cluster >= core_dim ||
      partition_ptr->n_per_cluster >= partition_ptr->h_per_cluster) {
    partition_ptr->core_partition = PARTITION_N;
    partition_ptr->n_per_core =
        (partition_ptr->n_per_cluster + core_dim - 1) / core_dim;
    partition_ptr->h_per_core = partition_ptr->h_per_cluster;
    use_core_num =
        (partition_ptr->n_per_cluster + partition_ptr->n_per_core - 1) /
        partition_ptr->n_per_core;
  } else {
    partition_ptr->core_partition = PARTITION_H;
    partition_ptr->h_per_core =
        (partition_ptr->h_per_cluster + core_dim - 1) / core_dim;
    partition_ptr->n_per_core = partition_ptr->n_per_cluster;
    use_core_num =
        (partition_ptr->h_per_cluster + partition_ptr->h_per_core - 1) /
        partition_ptr->h_per_core;
  }
  *k_dim_ptr = {core_dim, use_cluster_num, 1};
}

}  // namespace

bool findLimit(const int shape_core_n, const int shape_core_h,
               const int shape_core_w, const int shape_core_ci,
               const int shape_core_co, int *limit_n_seg_ptr,
               int *limit_h_seg_ptr, int *limit_w_seg_ptr, const int psa_type) {
  const bool need_temp = psa_type == 1;
  const int input_bytes = sizeof(float);
  int limit_n_seg = shape_core_n;
  int limit_h_seg = shape_core_h;
  int limit_w_seg = shape_core_w;

  const int max_nram_size = torch_mlu::getDeviceAttr(cnrtAttrNramSizePerMcore);
  const int align_base_128 = NFU_ALIGN_SIZE / input_bytes;
  const int align_base_64 = COMPUTE_COUNT_ALIGN / input_bytes;
  const int align_co = CEIL_ALIGN(shape_core_co, align_base_64);
  const int align_w = CEIL_ALIGN(shape_core_w, align_base_64);
  const int align_hw = CEIL_ALIGN(shape_core_h * shape_core_w, align_base_64);
  const int max_num = max_nram_size / input_bytes;

  int n_limit =
      max_num /
      (CEIL_ALIGN(shape_core_h * shape_core_w * shape_core_ci, align_base_128) +
       align_hw * align_co * (1 + need_temp));
  if (n_limit > 0) {
    n_limit = std::min(n_limit, shape_core_n);
    limit_n_seg = n_limit;
  } else {
    int h_limit =
        max_num / (CEIL_ALIGN(shape_core_w * shape_core_ci, align_base_128) +
                   align_w * align_co * (1 + need_temp));
    if (h_limit > 0) {
      h_limit = std::min(h_limit, shape_core_h);
      limit_h_seg = h_limit;
      limit_n_seg = 1;
    } else {
      int w_limit =
          max_num / (CEIL_ALIGN(shape_core_ci, align_base_128) +
                     CEIL_ALIGN(align_co, align_base_128) * (1 + need_temp));
      if (w_limit > 0 && w_limit >= (COMPUTE_COUNT_ALIGN / input_bytes)) {
        w_limit = std::min(w_limit, shape_core_w);
        w_limit = w_limit / (COMPUTE_COUNT_ALIGN / input_bytes) *
                  (COMPUTE_COUNT_ALIGN / input_bytes);
        limit_w_seg = w_limit;
        limit_h_seg = 1;
        limit_n_seg = 1;
      } else {
        CNLOG(INFO) << "The size of input channel is too large.";
        return false;
      }
    }
  }
  *limit_n_seg_ptr = limit_n_seg;
  *limit_h_seg_ptr = limit_h_seg;
  *limit_w_seg_ptr = limit_w_seg;
  return true;
}

void PSAMaskForwardMLUKernelLauncher(const int psa_type, const Tensor x,
                                     Tensor y, const int num_,
                                     const int h_feature, const int w_feature,
                                     const int h_mask, const int w_mask,
                                     const int half_h_mask,
                                     const int half_w_mask) {
  // params check
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x type should be Float, got ",
              x.scalar_type());
  TORCH_CHECK(y.scalar_type() == x.scalar_type(),
              "y should have the same type as x");
  TORCH_CHECK(x.dim() == 4, "x should be a 4d tensor, got ", x.dim(), "D");
  TORCH_CHECK(y.dim() == 4, "y should be a 4d tensor, got ", y.dim(), "D");

  int x_c = x.size(1);
  int y_c = y.size(1);
  TORCH_CHECK(h_mask * w_mask == x_c,
              "channel of x should be the same as h_mask * w_mask");
  TORCH_CHECK(h_feature * w_feature == y_c,
              "channel of y should be the same as h_feature * w_feature");
  TORCH_CHECK(psa_type == 0 || psa_type == 1,
              "psa_type only supports 'COLLECT' and 'DISTRIBUTE' currently");

  if (x.numel() == 0) {
    CNLOG(INFO) << "skip zero-element tensor";
    return;
  }

  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  PartitionSeg partition_info;
  policyFunc(&k_dim, &k_type, &partition_info, num_, h_feature);
  int n_limit_seg, h_limit_seg, w_limit_seg;
  bool ret =
      findLimit(partition_info.n_per_core, partition_info.h_per_core, w_feature,
                x_c, y_c, &n_limit_seg, &h_limit_seg, &w_limit_seg, psa_type);
  if (ret != true) {
    return;
  }

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(x.dim());
  auto x_tensor = torch_mlu::cnnl::ops::cnnl_contiguous(x, memory_format);
  at::Tensor y_tmp =
      at::empty({num_, y_c, h_feature, w_feature}, x.options(), memory_format);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto x_impl = torch_mlu::getMluTensorImpl(x_tensor);
  auto x_ptr = x_impl->cnnlMalloc();
  auto y_impl = torch_mlu::getMluTensorImpl(y_tmp);
  auto y_ptr = y_impl->cnnlMalloc();

  KernelPsamaskForward(
      k_dim, k_type, queue, x_ptr, y_ptr, (PsamaskType)psa_type,
      partition_info.core_partition, partition_info.cluster_partition, num_,
      h_feature, w_feature, h_mask, w_mask, x_c, y_c, half_h_mask, half_w_mask,
      partition_info.n_per_core, partition_info.h_per_core,
      partition_info.n_per_cluster, partition_info.h_per_cluster, n_limit_seg,
      h_limit_seg, w_limit_seg);

  y.copy_(y_tmp);
}

void PSAMaskBackwardMLUKernelLauncher(const int psa_type, const Tensor dy,
                                      Tensor dx, const int num_,
                                      const int h_feature, const int w_feature,
                                      const int h_mask, const int w_mask,
                                      const int half_h_mask,
                                      const int half_w_mask) {
  // params check
  TORCH_CHECK(dy.scalar_type() == at::kFloat, "dy type should be Float, got ",
              dy.scalar_type());
  TORCH_CHECK(dx.scalar_type() == dy.scalar_type(),
              "dx should have the same type as dy");
  TORCH_CHECK(dy.dim() == 4, "dy should be a 4d tensor, got ", dy.dim(), "D");
  TORCH_CHECK(dx.dim() == 4, "dx should be a 4d tensor, got ", dx.dim(), "D");

  int dy_c = dy.size(1);
  int dx_c = dx.size(1);
  TORCH_CHECK(h_feature * w_feature == dy_c,
              "channel of dy should be the same as h_feature * w_feature");
  TORCH_CHECK(h_mask * w_mask == dx_c,
              "channel of dx should be the same as h_mask * w_mask");
  TORCH_CHECK(psa_type == 0 || psa_type == 1,
              "psa_type only supports 'COLLECT' and 'DISTRIBUTE' currently");

  if (dx.numel() == 0) {
    CNLOG(INFO) << "skip zero-element tensor";
    return;
  }

  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  PartitionSeg partition_info;
  policyFunc(&k_dim, &k_type, &partition_info, num_, h_feature);
  int n_limit_seg, h_limit_seg, w_limit_seg;
  bool ret =
      findLimit(partition_info.n_per_core, partition_info.h_per_core, w_feature,
                dx_c, dy_c, &n_limit_seg, &h_limit_seg, &w_limit_seg, psa_type);
  if (ret != true) {
    return;
  }

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(dy.dim());
  auto dy_tensor = torch_mlu::cnnl::ops::cnnl_contiguous(dy, memory_format);
  at::Tensor dx_tmp = at::empty({num_, dx_c, h_feature, w_feature},
                                dy.options(), memory_format);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto dx_impl = torch_mlu::getMluTensorImpl(dx_tmp);
  auto dx_ptr = dx_impl->cnnlMalloc();
  auto dy_impl = torch_mlu::getMluTensorImpl(dy_tensor);
  auto dy_ptr = dy_impl->cnnlMalloc();

  KernelPsamaskBackward(
      k_dim, k_type, queue, dy_ptr, dx_ptr, (PsamaskType)psa_type,
      partition_info.core_partition, partition_info.cluster_partition, num_,
      h_feature, w_feature, h_mask, w_mask, dx_c, dy_c, half_h_mask,
      half_w_mask, partition_info.n_per_core, partition_info.h_per_core,
      partition_info.n_per_cluster, partition_info.h_per_cluster, n_limit_seg,
      h_limit_seg, w_limit_seg);

  dx.copy_(dx_tmp);
}

void psamask_forward_mlu(const int psa_type, const Tensor input, Tensor output,
                         const int num_, const int h_feature,
                         const int w_feature, const int h_mask,
                         const int w_mask, const int half_h_mask,
                         const int half_w_mask) {
  PSAMaskForwardMLUKernelLauncher(psa_type, input, output, num_, h_feature,
                                  w_feature, h_mask, w_mask, half_h_mask,
                                  half_w_mask);
}

void psamask_backward_mlu(const int psa_type, const Tensor grad_output,
                          Tensor grad_input, const int num_,
                          const int h_feature, const int w_feature,
                          const int h_mask, const int w_mask,
                          const int half_h_mask, const int half_w_mask) {
  PSAMaskBackwardMLUKernelLauncher(psa_type, grad_output, grad_input, num_,
                                   h_feature, w_feature, h_mask, w_mask,
                                   half_h_mask, half_w_mask);
}

void psamask_forward_impl(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask);

void psamask_backward_impl(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask);

REGISTER_DEVICE_IMPL(psamask_forward_impl, MLU, psamask_forward_mlu);
REGISTER_DEVICE_IMPL(psamask_backward_impl, MLU, psamask_backward_mlu);

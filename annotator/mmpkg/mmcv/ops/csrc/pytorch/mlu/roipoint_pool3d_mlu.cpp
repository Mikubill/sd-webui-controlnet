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

void KernelRoiPointPool3dForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                 cnrtQueue_t queue, const cnrtDataType_t d_type,
                                 const int batch_size, const int pts_num,
                                 const int boxes_num, const int feature_in_len,
                                 const int sampled_pts_num, const void *xyz,
                                 const void *boxes3d, const void *pts_feature,
                                 void *pooled_features, int *pooled_empty_flag);

void KernelRoiPointPool3dLargeBoxesNumForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const int batch_size, const int pts_num,
    const int boxes_num, const int feature_in_len, const int sampled_pts_num,
    const void *xyz, const void *boxes3d, const void *pts_feature,
    void *pooled_features, int *pooled_empty_flag);

// policy function
static void policyFuncForward(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  // start U1 task, occupy all available clusters
  k_dim->x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

void RoIPointPool3dForwardMLUKernelLauncher(
    int batch_size, int pts_num, int boxes_num, int feature_in_len,
    int sampled_pts_num, const Tensor xyz, const Tensor boxes3d,
    const Tensor pts_feature, Tensor pooled_features,
    Tensor pooled_empty_flag) {
  // check datatype
  TORCH_CHECK(((xyz.scalar_type() == pooled_features.scalar_type()) &&
               (boxes3d.scalar_type() == pooled_features.scalar_type()) &&
               (pts_feature.scalar_type() == pooled_features.scalar_type())),
              "data types of xyz, boxes3d, pts_feature and pooled_features "
              "should be the same, ",
              "but now xyz type is ", xyz.scalar_type(), ", boxes3d type is ",
              boxes3d.scalar_type(), ", pts_feature type is ",
              pts_feature.scalar_type(), ", pooled_features type is ",
              pooled_features.scalar_type(), ".");
  TORCH_CHECK(
      (xyz.scalar_type() == at::kFloat || xyz.scalar_type() == at::kHalf),
      "xyz type should be Float or Half, got ", xyz.scalar_type(), ".");
  TORCH_CHECK((pooled_empty_flag.scalar_type() == at::kInt),
              "pooled_empty_flag type should be Int, got ",
              pooled_empty_flag.scalar_type(), ".");

  // check shape
  TORCH_CHECK(boxes3d.dim() == 3, "boxes3d should be a 3d tensor, got ",
              boxes3d.dim(), "D.");
  TORCH_CHECK(pts_feature.dim() == 3, "pts_feature should be a 3d tensor, got ",
              pts_feature.dim(), "D.");

  TORCH_CHECK(boxes3d.size(2) == 7,
              "the 3rd dimensions of boxes3d should be 7, got ",
              boxes3d.size(2), ".");
  TORCH_CHECK((boxes3d.size(0) == batch_size),
              "the 1st dimensions of boxes3d should be batch_size, ",
              "but now the 1st dimension of boxes3d is ", boxes3d.size(0),
              ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((pts_feature.size(0) == batch_size),
              "the 1st dimensions of pts_feature should be batch_size, ",
              "but now the 1st dimension of pts_feature is ",
              pts_feature.size(0), ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((pts_feature.size(1) == pts_num),
              "the 2nd dimensions of pts_feature should be pts_num, ",
              "but now the 2nd dimension of pts_feature is ",
              pts_feature.size(1), ", and pts_num is ", pts_num, ".");

  // check zero element
  if (xyz.numel() == 0 || pts_feature.numel() == 0 || boxes3d.numel() == 0 ||
      pooled_features.numel() == 0 || pooled_empty_flag.numel() == 0) {
    return;
  }

  // large tensor check
  const size_t max_input_size = 2147483648;
  TORCH_CHECK(xyz.numel() < max_input_size,
              "xyz element num should be less than 2^31, got ", xyz.numel(),
              ".");
  TORCH_CHECK(boxes3d.numel() < max_input_size,
              "boxes3d element num should be less than 2^31, got ",
              boxes3d.numel(), ".");
  TORCH_CHECK(pts_feature.numel() < max_input_size,
              "pts_feature element num should be less than 2^31, got ",
              pts_feature.numel(), ".");

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncForward(&k_dim, &k_type);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  // transpose points [B, N ,3] -> [3, B, N]
  auto xyz_ = xyz.permute({2, 0, 1}).contiguous();
  auto xyz_impl = torch_mlu::getMluTensorImpl(xyz_);
  auto xyz_ptr = xyz_impl->cnnlMalloc();
  // transpose point_features [B, N, C] -> [B, C, N]
  auto pts_feature_ = pts_feature.permute({0, 2, 1}).contiguous();
  auto pts_feature_impl = torch_mlu::getMluTensorImpl(pts_feature_);
  auto pts_feature_ptr = pts_feature_impl->cnnlMalloc();
  auto boxes3d_impl = torch_mlu::getMluTensorImpl(boxes3d);
  auto boxes3d_ptr = boxes3d_impl->cnnlMalloc();
  auto pooled_features_impl = torch_mlu::getMluTensorImpl(pooled_features);
  auto pooled_features_ptr = pooled_features_impl->cnnlMalloc();
  auto pooled_empty_flag_impl = torch_mlu::getMluTensorImpl(pooled_empty_flag);
  auto pooled_empty_flag_ptr = pooled_empty_flag_impl->cnnlMalloc();

  // get compute dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(xyz_.dtype());

  // launch kernel
  if (boxes_num <= 10240) {
    CNLOG(INFO) << "Launch Kernel MLUKernelRoiPointPool3dForward<<<" << k_dim.x
                << ", " << k_dim.y << ", " << k_dim.z << ">>>";
    KernelRoiPointPool3dForward(
        k_dim, k_type, queue, data_type, batch_size, pts_num, boxes_num,
        feature_in_len, sampled_pts_num, xyz_ptr, boxes3d_ptr, pts_feature_ptr,
        pooled_features_ptr, (int *)pooled_empty_flag_ptr);
  } else {
    CNLOG(INFO)
        << "Launch Kernel MLUKernelRoiPointPool3dLargeBoxesNumForward<<<"
        << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
    KernelRoiPointPool3dLargeBoxesNumForward(
        k_dim, k_type, queue, data_type, batch_size, pts_num, boxes_num,
        feature_in_len, sampled_pts_num, xyz_ptr, boxes3d_ptr, pts_feature_ptr,
        pooled_features_ptr, (int *)pooled_empty_flag_ptr);
  }
}

void roipoint_pool3d_forward_mlu(int batch_size, int pts_num, int boxes_num,
                                 int feature_in_len, int sampled_pts_num,
                                 const Tensor xyz, const Tensor boxes3d,
                                 const Tensor pts_feature,
                                 Tensor pooled_features,
                                 Tensor pooled_empty_flag) {
  RoIPointPool3dForwardMLUKernelLauncher(
      batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num, xyz,
      boxes3d, pts_feature, pooled_features, pooled_empty_flag);
}

void roipoint_pool3d_forward_impl(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const Tensor xyz, const Tensor boxes3d,
                                  const Tensor pts_feature,
                                  Tensor pooled_features,
                                  Tensor pooled_empty_flag);

REGISTER_DEVICE_IMPL(roipoint_pool3d_forward_impl, MLU,
                     roipoint_pool3d_forward_mlu);

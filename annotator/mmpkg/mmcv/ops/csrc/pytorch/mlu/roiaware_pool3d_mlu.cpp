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

void KernelPtsIdxOfVoxels(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                          cnrtQueue_t queue, const cnrtDataType_t d_type,
                          const int pool_method, const int boxes_num,
                          const int pts_num, const int max_pts_each_voxel,
                          const int out_x, const int out_y, const int out_z,
                          const void *rois, const void *pts,
                          int *pts_idx_of_voxels);

void KernelRoiawarePool3dForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const int pool_method, const int boxes_num,
    const int pts_num, const int channels, const int max_pts_each_voxel,
    const int out_x, const int out_y, const int out_z, const void *pts_feature,
    const int *pts_idx_of_voxels, void *pooled_features, int *argmax);

// policy function
static void kernelPtsIdxOfVoxelsPolicyFunc(const int boxes_num,
                                           cnrtDim3_t *k_dim,
                                           cnrtFunctionType_t *k_type) {
  unsigned int core_num = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  unsigned int cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_num;
  unsigned int use_cluster = (boxes_num + core_num - 1) / core_num;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

static void kernelRoiawarePool3dForwardPolicyFunc(
    const int boxes_num, const int out_x, const int out_y, const int out_z,
    cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  unsigned int core_num = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  unsigned int cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_num;
  const int voxels_num = boxes_num * out_x * out_y * out_z;
  unsigned int use_cluster = (voxels_num + core_num - 1) / core_num;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

void RoiawarePool3dForwardMLUKernelLauncher(
    const int pool_method, const int boxes_num, const int pts_num,
    const int channels, const int max_pts_each_voxel, const int out_x,
    const int out_y, const int out_z, const Tensor rois, const Tensor pts,
    const Tensor pts_feature, Tensor pts_idx_of_voxels, Tensor pooled_features,
    Tensor argmax) {
  // check datatype
  TORCH_CHECK(((pts.scalar_type() == rois.scalar_type()) &&
               (pts_feature.scalar_type() == rois.scalar_type()) &&
               (pooled_features.scalar_type() == rois.scalar_type())),
              "data types of rois, rois, pts_feature and pooled_features "
              "should be the same, ",
              "but now rois type is ", rois.scalar_type(), ", pts type is ",
              pts.scalar_type(), ", pts_feature type is ",
              pts_feature.scalar_type(), ", pooled_features type is ",
              pooled_features.scalar_type(), ".");
  TORCH_CHECK(
      (rois.scalar_type() == at::kFloat || rois.scalar_type() == at::kHalf),
      "rois type should be Float or Half, got ", rois.scalar_type(), ".");
  TORCH_CHECK((pts_idx_of_voxels.scalar_type() == at::kInt),
              "pts_idx_of_voxels type should be Int, got ",
              pts_idx_of_voxels.scalar_type(), ".");
  // check dim
  TORCH_CHECK(rois.dim() == 2, "rois should be a 2D tensor, got ", rois.dim(),
              "D.");
  TORCH_CHECK(pts.dim() == 2, "pts should be a 2D tensor, got ", pts.dim(),
              "D.");
  TORCH_CHECK(pts_feature.dim() == 2, "pts_feature should be a 2D tensor, got ",
              pts_feature.dim(), "D.");
  TORCH_CHECK(pts_idx_of_voxels.dim() == 5,
              "pts_idx_of_voxels should be a 5D tensor, got ",
              pts_idx_of_voxels.dim(), "D.");
  TORCH_CHECK(pooled_features.dim() == 5,
              "pooled_features should be a 5D tensor, got ",
              pooled_features.dim(), "D.");
  // check shape
  TORCH_CHECK(((rois.size(0) == boxes_num) && (rois.size(1) == 7)),
              "the dimensions of rois should be (boxes_num, 7), ", "but got (",
              rois.size(0), ", ", rois.size(1), ") .");
  TORCH_CHECK(((pts.size(0) == pts_num) && (pts.size(1) == 3)),
              "the dimensions of pts should be (pts_num, 3), ", "but got (",
              pts.size(0), ",", pts.size(1), ").");
  TORCH_CHECK(
      ((pts_feature.size(0) == pts_num) && (pts_feature.size(1) == channels)),
      "the dimensions of pts_feature should be (pts_num, channels), ",
      "but got (", pts_feature.size(0), ",", pts_feature.size(1), ").");
  TORCH_CHECK(((pts_idx_of_voxels.size(0) == boxes_num) &&
               (pts_idx_of_voxels.size(1) == out_x) &&
               (pts_idx_of_voxels.size(2) == out_y) &&
               (pts_idx_of_voxels.size(3) == out_z) &&
               (pts_idx_of_voxels.size(4) == max_pts_each_voxel)),
              "the dimensions of pts_idx_of_voxels should be (boxes_num, "
              "out_x, out_y, out_z, max_pts_each_voxel), ",
              "but got (", pts_idx_of_voxels.size(0), ",",
              pts_idx_of_voxels.size(1), ",", pts_idx_of_voxels.size(2), ",",
              pts_idx_of_voxels.size(3), ",", pts_idx_of_voxels.size(4), ").");
  TORCH_CHECK(((pooled_features.size(0) == boxes_num) &&
               (pooled_features.size(1) == out_x) &&
               (pooled_features.size(2) == out_y) &&
               (pooled_features.size(3) == out_z) &&
               (pooled_features.size(4) == channels)),
              "the dimensions of pooled_features should be (boxes_num, out_x, "
              "out_y, out_z, channels), ",
              "but got (", pooled_features.size(0), ",",
              pooled_features.size(1), ",", pooled_features.size(2), ",",
              pooled_features.size(3), ",", pooled_features.size(4), ").");
  // check other params : pool_mothod
  TORCH_CHECK(((pool_method == 0) || (pool_method == 1)),
              "the num of pool_method should be 0(max) or 1(avg), ", "but got ",
              pool_method, ".");
  // check large tensor
  const size_t max_input_size = 2147483648;
  TORCH_CHECK(rois.numel() < max_input_size,
              "rois element num should be less than 2^31, got ", rois.numel(),
              ".");
  TORCH_CHECK(pts.numel() < max_input_size,
              "pts element num should be less than 2^31, got ", pts.numel(),
              ".");
  TORCH_CHECK(pts_feature.numel() < max_input_size,
              "pts_feature element num should be less than 2^31, got ",
              pts_feature.numel(), ".");
  TORCH_CHECK(pts_idx_of_voxels.numel() < max_input_size,
              "pts_idx_of_voxels element num should be less than 2^31, got ",
              pts_idx_of_voxels.numel(), ".");
  TORCH_CHECK(pooled_features.numel() < max_input_size,
              "pooled_features element num should be less than 2^31, got ",
              pooled_features.numel(), ".");
  // check zero element
  TORCH_CHECK(rois.numel() != 0, "rois.numel() should not be zero, got ",
              rois.numel());
  TORCH_CHECK(pts.numel() != 0, "pts.numel() should not be zero, got ",
              pts.numel());
  TORCH_CHECK(pts_feature.numel() != 0,
              "pts_feature.numel() should not be zero, got ",
              pts_feature.numel());
  TORCH_CHECK(pts_idx_of_voxels.numel() != 0,
              "pts_idx_of_voxels.numel() should not be zero, got ",
              pts_idx_of_voxels.numel());
  TORCH_CHECK(pooled_features.numel() != 0,
              "pooled_features.numel() should not be zero, got ",
              pooled_features.numel());
  if (pool_method == 0) {
    // check datatype
    TORCH_CHECK((argmax.scalar_type() == at::kInt),
                "argmax type should be Int, got ", argmax.scalar_type(), ".");
    // check dim
    TORCH_CHECK(argmax.dim() == 5, "argmax should be a 5D tensor, got ",
                argmax.dim(), "D.");
    // check shape
    TORCH_CHECK(((argmax.size(0) == boxes_num) && (argmax.size(1) == out_x) &&
                 (argmax.size(2) == out_y) && (argmax.size(3) == out_z) &&
                 (argmax.size(4) == channels)),
                "the dimensions of argmax should be (boxes_num, out_x, out_y, "
                "out_z, channels), ",
                "but got (", argmax.size(0), ",", argmax.size(1), ",",
                argmax.size(2), ",", argmax.size(3), ",", argmax.size(4), ").");
    // check large tensor
    TORCH_CHECK(argmax.numel() < max_input_size,
                "argmax element num should be less than 2^31, got ",
                argmax.numel(), ".");
    // check zero element
    TORCH_CHECK(argmax.numel() != 0, "argmax.numel() should not be zero, got ",
                argmax.numel());
    // when pool_method is 0, which is max pool, init argmax data value to -1
    argmax.fill_(static_cast<int>(-1));
  }
  // calculate task one dimension
  cnrtDim3_t k1_dim;
  cnrtFunctionType_t k1_type;
  kernelPtsIdxOfVoxelsPolicyFunc(boxes_num, &k1_dim, &k1_type);
  cnrtDim3_t k2_dim;
  cnrtFunctionType_t k2_type;
  kernelRoiawarePool3dForwardPolicyFunc(boxes_num, out_x, out_y, out_z, &k2_dim,
                                        &k2_type);
  // get compute queue
  auto queue = torch_mlu::getCurQueue();
  // get ptr of tensors
  auto rois_impl = torch_mlu::getMluTensorImpl(rois);
  auto rois_ptr = rois_impl->cnnlMalloc();
  // transpose points [pts_num, 3] -> [3, pts_num]
  auto pts_ = pts.permute({1, 0}).contiguous();
  auto pts_impl = torch_mlu::getMluTensorImpl(pts_);
  auto pts_ptr = pts_impl->cnnlMalloc();
  // transpose points_features [pts_num, channels] -> [channels, pts_num]
  auto pts_feature_ = pts_feature.permute({1, 0}).contiguous();
  auto pts_feature_impl = torch_mlu::getMluTensorImpl(pts_feature_);
  auto pts_feature_ptr = pts_feature_impl->cnnlMalloc();
  auto pts_idx_of_voxels_impl = torch_mlu::getMluTensorImpl(pts_idx_of_voxels);
  auto pts_idx_of_voxels_ptr = pts_idx_of_voxels_impl->cnnlMalloc();
  auto pooled_features_impl = torch_mlu::getMluTensorImpl(pooled_features);
  auto pooled_features_ptr = pooled_features_impl->cnnlMalloc();
  auto argmax_impl = torch_mlu::getMluTensorImpl(argmax);
  auto argmax_ptr = argmax_impl->cnnlMalloc();
  // get compute dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(rois.dtype());
  // launch kernel PtsIdxOfVoxels
  CNLOG(INFO) << "Launch Kernel MLUKernel PtsIdxOfVoxels<<<" << k1_dim.x << ", "
              << k1_dim.y << ", " << k1_dim.z << ">>>";
  KernelPtsIdxOfVoxels(k1_dim, k1_type, queue, data_type, pool_method,
                       boxes_num, pts_num, max_pts_each_voxel, out_x, out_y,
                       out_z, rois_ptr, pts_ptr, (int *)pts_idx_of_voxels_ptr);
  // launch kernel RoiawarePool3dForward
  CNLOG(INFO) << "Launch Kernel MLUKernel RoiawarePool3dForward<<<" << k2_dim.x
              << ", " << k2_dim.y << ", " << k2_dim.z << ">>>";
  KernelRoiawarePool3dForward(
      k2_dim, k2_type, queue, data_type, pool_method, boxes_num, pts_num,
      channels, max_pts_each_voxel, out_x, out_y, out_z, pts_feature_ptr,
      (int *)pts_idx_of_voxels_ptr, pooled_features_ptr, (int *)argmax_ptr);
}

void roiaware_pool3d_forward_mlu(int boxes_num, int pts_num, int channels,
                                 int max_pts_each_voxel, int out_x, int out_y,
                                 int out_z, const Tensor rois, const Tensor pts,
                                 const Tensor pts_feature, Tensor argmax,
                                 Tensor pts_idx_of_voxels,
                                 Tensor pooled_features, int pool_method) {
  RoiawarePool3dForwardMLUKernelLauncher(
      pool_method, boxes_num, pts_num, channels, max_pts_each_voxel, out_x,
      out_y, out_z, rois, pts, pts_feature, pts_idx_of_voxels, pooled_features,
      argmax);
}

void roiaware_pool3d_forward_impl(int boxes_num, int pts_num, int channels,
                                  int max_pts_each_voxel, int out_x, int out_y,
                                  int out_z, const Tensor rois,
                                  const Tensor pts, const Tensor pts_feature,
                                  Tensor argmax, Tensor pts_idx_of_voxels,
                                  Tensor pooled_features, int pool_method);

REGISTER_DEVICE_IMPL(roiaware_pool3d_forward_impl, MLU,
                     roiaware_pool3d_forward_mlu);

void KernelRoiawarePool3dBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const int pool_method, const int boxes_num,
    const int out_x, const int out_y, const int out_z, const int channels,
    const int max_pts_each_voxel, const int *pts_idx_of_voxels,
    const int *argmax, const void *grad_out, void *grad_in);

static void kernelRoiawarePool3dBackwardPolicyFunc(
    const int boxes_num, const int out_x, const int out_y, const int out_z,
    cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  unsigned int core_num = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  unsigned int cluster_num = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_num;
  const int voxels_num = boxes_num * out_x * out_y * out_z;
  unsigned int use_cluster = (voxels_num + core_num - 1) / core_num;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

void RoiawarePool3dBackwardMLUKernelLauncher(
    int pool_method, int boxes_num, int out_x, int out_y, int out_z,
    int channels, int max_pts_each_voxel, const Tensor pts_idx_of_voxels,
    const Tensor argmax, const Tensor grad_out, Tensor grad_in) {
  // check datatype
  TORCH_CHECK((pts_idx_of_voxels.scalar_type() == at::kInt),
              "pts_idx_of_voxels type should be Int, got ",
              pts_idx_of_voxels.scalar_type(), ".");
  TORCH_CHECK((argmax.scalar_type() == at::kInt),
              "argmax type should be Int, got ", argmax.scalar_type(), ".");
  TORCH_CHECK((grad_out.scalar_type() == at::kFloat ||
               grad_out.scalar_type() == at::kHalf),
              "grad_out type should be Float or Half, got ",
              grad_out.scalar_type(), ".");
  TORCH_CHECK((grad_out.scalar_type() == grad_in.scalar_type()),
              "data types of grad_out, grad_in, should be the same, ",
              "but now grad_out type is ", grad_out.scalar_type(),
              ", grad_in type is ", grad_in.scalar_type(), ".");
  // check dim
  TORCH_CHECK(pts_idx_of_voxels.dim() == 5,
              "pts_idx_of_voxels should be a 5D tensor, got ",
              pts_idx_of_voxels.dim(), "D.");
  TORCH_CHECK(argmax.dim() == 5, "argmax should be a 5D tensor, got ",
              argmax.dim(), "D.");
  TORCH_CHECK(grad_out.dim() == 5, "grad_out should be a 5D tensor, got ",
              grad_out.dim(), "D.");
  TORCH_CHECK(grad_in.dim() == 2, "grad_in should be a 2D tensor, got ",
              grad_in.dim(), "D.");
  // check shape
  TORCH_CHECK(((pts_idx_of_voxels.size(0) == boxes_num) &&
               (pts_idx_of_voxels.size(1) == out_x) &&
               (pts_idx_of_voxels.size(2) == out_y) &&
               (pts_idx_of_voxels.size(3) == out_z) &&
               (pts_idx_of_voxels.size(4) == max_pts_each_voxel)),
              "the dimensions of pts_idx_of_voxels should be (boxes_num, "
              "out_x, out_y, out_z, max_pts_each_voxel), ",
              "but got (", pts_idx_of_voxels.size(0), ",",
              pts_idx_of_voxels.size(1), ",", pts_idx_of_voxels.size(2), ",",
              pts_idx_of_voxels.size(3), ",", pts_idx_of_voxels.size(4), ").");
  TORCH_CHECK(((argmax.size(0) == boxes_num) && (argmax.size(1) == out_x) &&
               (argmax.size(2) == out_y) && (argmax.size(3) == out_z) &&
               (argmax.size(4) == channels)),
              "the dimensions of argmax should be (boxes_num, out_x, out_y, "
              "out_z, channels), ",
              "but got (", argmax.size(0), ",", argmax.size(1), ",",
              argmax.size(2), ",", argmax.size(3), ",", argmax.size(4), ").");
  TORCH_CHECK(((grad_out.size(0) == boxes_num) && (grad_out.size(1) == out_x) &&
               (grad_out.size(2) == out_y) && (grad_out.size(3) == out_z) &&
               (grad_out.size(4) == channels)),
              "the dimensions of grad_out should be (boxes_num, out_x, "
              "out_y, out_z, channels), ",
              "but got (", grad_out.size(0), ",", grad_out.size(1), ",",
              grad_out.size(2), ",", grad_out.size(3), ",", grad_out.size(4),
              ").");
  TORCH_CHECK((grad_in.size(1) == channels),
              "the 1st dimensions of grad_in should be channels, ", "but got ",
              grad_in.size(1), ".");
  // check other params : pool_mothod
  TORCH_CHECK(((pool_method == 0) || (pool_method == 1)),
              "the num of pool_method should be 0(max) or 1(avg), ", "but got ",
              pool_method, ".");
  // check large tensor
  const size_t max_input_size = 2147483648;
  TORCH_CHECK(pts_idx_of_voxels.numel() < max_input_size,
              "pts_idx_of_voxels element num should be less than 2^31, got ",
              pts_idx_of_voxels.numel(), ".");
  TORCH_CHECK(argmax.numel() < max_input_size,
              "argmax element num should be less than 2^31, got ",
              argmax.numel(), ".");
  TORCH_CHECK(grad_out.numel() < max_input_size,
              "grad_out element num should be less than 2^31, got ",
              grad_out.numel(), ".");
  TORCH_CHECK(grad_in.numel() < max_input_size,
              "grad_in element num should be less than 2^31, got ",
              grad_in.numel(), ".");
  // check zero element
  TORCH_CHECK(pts_idx_of_voxels.numel() != 0,
              "pts_idx_of_voxels.numel() should not be zero, got ",
              pts_idx_of_voxels.numel());
  TORCH_CHECK(argmax.numel() != 0, "argmax.numel() should not be zero, got ",
              argmax.numel());
  TORCH_CHECK(grad_out.numel() != 0,
              "grad_out.numel() should not be zero, got ", grad_out.numel());
  TORCH_CHECK(grad_in.numel() != 0, "grad_in.numel() should not be zero, got ",
              grad_in.numel());
  // calculate task one dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  kernelRoiawarePool3dBackwardPolicyFunc(boxes_num, out_x, out_y, out_z, &k_dim,
                                         &k_type);
  // get compute queue
  auto queue = torch_mlu::getCurQueue();
  // transpose points_features [pts_num, channels] -> [channels, pts_num]
  auto pts_idx_of_voxels_impl = torch_mlu::getMluTensorImpl(pts_idx_of_voxels);
  auto pts_idx_of_voxels_ptr = pts_idx_of_voxels_impl->cnnlMalloc();
  auto argmax_impl = torch_mlu::getMluTensorImpl(argmax);
  auto argmax_ptr = argmax_impl->cnnlMalloc();
  auto grad_out_impl = torch_mlu::getMluTensorImpl(grad_out);
  auto grad_out_ptr = grad_out_impl->cnnlMalloc();
  auto grad_in_impl = torch_mlu::getMluTensorImpl(grad_in);
  auto grad_in_ptr = grad_in_impl->cnnlMalloc();
  // get compute dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(grad_out.dtype());
  // launch kernel RoiawarePool3dForward
  CNLOG(INFO) << "Launch Kernel MLUKernel RoiawarePool3dBackward<<<" << k_dim.x
              << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  KernelRoiawarePool3dBackward(k_dim, k_type, queue, data_type, pool_method,
                               boxes_num, out_x, out_y, out_z, channels,
                               max_pts_each_voxel, (int *)pts_idx_of_voxels_ptr,
                               (int *)argmax_ptr, grad_out_ptr, grad_in_ptr);
}

void roiaware_pool3d_backward_mlu(int boxes_num, int out_x, int out_y,
                                  int out_z, int channels,
                                  int max_pts_each_voxel,
                                  const Tensor pts_idx_of_voxels,
                                  const Tensor argmax, const Tensor grad_out,
                                  Tensor grad_in, int pool_method) {
  RoiawarePool3dBackwardMLUKernelLauncher(
      pool_method, boxes_num, out_x, out_y, out_z, channels, max_pts_each_voxel,
      pts_idx_of_voxels, argmax, grad_out, grad_in);
}

void roiaware_pool3d_backward_impl(int boxes_num, int out_x, int out_y,
                                   int out_z, int channels,
                                   int max_pts_each_voxel,
                                   const Tensor pts_idx_of_voxels,
                                   const Tensor argmax, const Tensor grad_out,
                                   Tensor grad_in, int pool_method);

REGISTER_DEVICE_IMPL(roiaware_pool3d_backward_impl, MLU,
                     roiaware_pool3d_backward_mlu);

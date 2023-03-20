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

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void KernelDynamicVoxelize(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *points, void *coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int32_t grid_x, const int32_t grid_y,
    const int32_t grid_z, const int32_t num_points, const int32_t num_features);

void KernelPoint2Voxel(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                       cnrtQueue_t queue, void *coors, void *point_to_pointidx,
                       void *point_to_voxelidx, const int32_t num_points,
                       const int32_t max_points);

void KernelCalcPointsPerVoxel(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                              cnrtQueue_t queue, void *point_to_pointidx,
                              void *point_to_voxelidx, void *coor_to_voxelidx,
                              void *num_points_per_voxel, void *voxel_num,
                              const int32_t max_voxels,
                              const int32_t num_points);

void KernelAssignVoxelsCoors(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                             cnrtQueue_t queue, const void *points,
                             void *temp_coors, void *point_to_voxelidx,
                             void *coor_to_voxelidx, void *voxels, void *coors,
                             const int32_t max_points, const int32_t num_points,
                             const int32_t num_features);

// policy function
static void policyFuncDefault(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                              const int num_points) {
  k_dim->x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = MIN((num_points + k_dim->x - 1) / k_dim->x,
                 torch_mlu::getDeviceAttr(cnrtAttrClusterCount));
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

// policy function
static void policyFuncCalcPointsPerVoxel(cnrtDim3_t *k_dim,
                                         cnrtFunctionType_t *k_type,
                                         const int num_points) {
  k_dim->x = 1;
  k_dim->y = 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_BLOCK;
}

int HardVoxelizeForwardMLUKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  // check datatype
  TORCH_CHECK(points.scalar_type() == at::kFloat,
              "points type should be Float, got ", points.scalar_type(), ".");
  TORCH_CHECK(voxels.scalar_type() == at::kFloat,
              "voxels type should be Float, got ", voxels.scalar_type(), ".");
  TORCH_CHECK(coors.scalar_type() == at::kInt,
              "coors type should be Float, got ", coors.scalar_type(), ".");
  TORCH_CHECK(num_points_per_voxel.scalar_type() == at::kInt,
              "num_points_per_voxel type should be Float, got ",
              num_points_per_voxel.scalar_type(), ".");

  // check shape
  TORCH_CHECK(points.dim() == 2, "points should be a 2d tensor, got ",
              points.dim(), "D.");
  TORCH_CHECK(voxels.dim() == 3, "voxels should be a 3d tensor, got ",
              voxels.dim(), "D.");
  TORCH_CHECK(coors.dim() == 2, "coors should be a 2d tensor, got ",
              coors.dim(), "D.");
  TORCH_CHECK(num_points_per_voxel.dim() == 1,
              "num_points_per_voxel should be a 1d tensor, got ",
              num_points_per_voxel.dim(), "D.");

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  TORCH_CHECK(points.size(0) == num_points,
              "the 1st dimensions of points should be num_points, got ",
              points.size(0), ".");
  TORCH_CHECK(points.size(1) == num_features,
              "the 2nd dimensions of points should be num_features, got ",
              points.size(1), ".");
  TORCH_CHECK(voxels.size(0) == max_voxels,
              "the 1st dimensions of voxels should be max_voxels, got ",
              voxels.size(0), ".");
  TORCH_CHECK(voxels.size(1) == max_points,
              "the 2nd dimensions of voxels should be max_points, got ",
              voxels.size(1), ".");
  TORCH_CHECK(voxels.size(2) == num_features,
              "the 3rd dimensions of voxels should be num_features, got ",
              voxels.size(2), ".");
  TORCH_CHECK(coors.size(0) == max_voxels,
              "the 1st dimensions of coors should be max_voxels, got ",
              coors.size(0), ".");
  TORCH_CHECK(coors.size(1) == 3,
              "the 2nd dimensions of coors should be 3, got ", coors.size(1),
              ".");
  TORCH_CHECK(num_points_per_voxel.size(0) == max_voxels,
              "the 1st dimensions of num_points_per_voxel should be 3, got ",
              num_points_per_voxel.size(0), ".");

  // large tensor check
  const size_t max_input_size = 2147483648;
  TORCH_CHECK(points.numel() < max_input_size,
              "points element num should be less than 2^31, got ",
              points.numel(), ".");
  TORCH_CHECK(voxels.numel() < max_input_size,
              "voxels element num should be less than 2^31, got ",
              voxels.numel(), ".");
  TORCH_CHECK(coors.numel() < max_input_size,
              "coors element num should be less than 2^31, got ", coors.numel(),
              ".");

  // check zero element
  if (max_points == 0 || max_voxels == 0) {
    return 0;
  }

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto points_ = points.contiguous();
  auto points_impl = torch_mlu::getMluTensorImpl(points_);
  auto points_ptr = points_impl->cnnlMalloc();
  auto voxels_ = voxels.contiguous();
  auto voxels_impl = torch_mlu::getMluTensorImpl(voxels_);
  auto voxels_ptr = voxels_impl->cnnlMalloc();
  auto coors_ = coors.contiguous();
  auto coors_impl = torch_mlu::getMluTensorImpl(coors_);
  auto coors_ptr = coors_impl->cnnlMalloc();
  auto num_points_per_voxel_ = num_points_per_voxel.contiguous();
  auto num_points_per_voxel_impl =
      torch_mlu::getMluTensorImpl(num_points_per_voxel_);
  auto num_points_per_voxel_ptr = num_points_per_voxel_impl->cnnlMalloc();

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncDefault(&k_dim, &k_type, num_points);

  // 1. link point to corresponding voxel coors
  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  auto temp_coors =
      at::zeros({NDim, num_points}, points.options().dtype(at::kInt))
          .contiguous();
  auto temp_coors_impl = torch_mlu::getMluTensorImpl(temp_coors);
  auto temp_coors_ptr = temp_coors_impl->cnnlMalloc();

  KernelDynamicVoxelize(k_dim, k_type, queue, points_ptr, temp_coors_ptr,
                        voxel_x, voxel_y, voxel_z, coors_x_min, coors_y_min,
                        coors_z_min, coors_x_max, coors_y_max, coors_z_max,
                        grid_x, grid_y, grid_z, num_points, num_features);

  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  auto point_to_pointidx = at::zeros(
                               {
                                   num_points,
                               },
                               points.options().dtype(at::kInt))
                               .contiguous();
  auto point_to_pointidx_impl = torch_mlu::getMluTensorImpl(point_to_pointidx);
  auto point_to_pointidx_ptr = point_to_pointidx_impl->cnnlMalloc();
  auto point_to_voxelidx = at::zeros(
                               {
                                   num_points,
                               },
                               points.options().dtype(at::kInt))
                               .contiguous();
  auto point_to_voxelidx_impl = torch_mlu::getMluTensorImpl(point_to_voxelidx);
  auto point_to_voxelidx_ptr = point_to_voxelidx_impl->cnnlMalloc();

  KernelPoint2Voxel(k_dim, k_type, queue, temp_coors_ptr, point_to_pointidx_ptr,
                    point_to_voxelidx_ptr, num_points, max_points);

  // calculate task dimension
  cnrtDim3_t k_dim_calc_points_per_voxel;
  cnrtFunctionType_t k_type_calc_points_per_voxel;
  policyFuncCalcPointsPerVoxel(&k_dim_calc_points_per_voxel,
                               &k_type_calc_points_per_voxel, num_points);

  // 3. determine voxel num and voxel's coor index
  auto coor_to_voxelidx = at::zeros(
                              {
                                  num_points,
                              },
                              points.options().dtype(at::kInt))
                              .contiguous();
  auto coor_to_voxelidx_impl = torch_mlu::getMluTensorImpl(coor_to_voxelidx);
  auto coor_to_voxelidx_ptr = coor_to_voxelidx_impl->cnnlMalloc();
  auto voxel_num = at::zeros(
                       {
                           1,
                       },
                       points.options().dtype(at::kInt))
                       .contiguous();
  auto voxel_num_impl = torch_mlu::getMluTensorImpl(voxel_num);
  auto voxel_num_ptr = voxel_num_impl->cnnlMalloc();

  KernelCalcPointsPerVoxel(
      k_dim_calc_points_per_voxel, k_type_calc_points_per_voxel, queue,
      point_to_pointidx_ptr, point_to_voxelidx_ptr, coor_to_voxelidx_ptr,
      num_points_per_voxel_ptr, voxel_num_ptr, max_voxels, num_points);

  // 4. copy point features and coors of each voxels to voxels
  KernelAssignVoxelsCoors(k_dim, k_type, queue, points_ptr, temp_coors_ptr,
                          point_to_voxelidx_ptr, coor_to_voxelidx_ptr,
                          voxels_ptr, coors_ptr, max_points, num_points,
                          num_features);

  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];

  return voxel_num_int;
}

int hard_voxelize_forward_mlu(const at::Tensor &points, at::Tensor &voxels,
                              at::Tensor &coors,
                              at::Tensor &num_points_per_voxel,
                              const std::vector<float> voxel_size,
                              const std::vector<float> coors_range,
                              const int max_points, const int max_voxels,
                              const int NDim) {
  return HardVoxelizeForwardMLUKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size, coors_range,
      max_points, max_voxels, NDim);
};

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim);

REGISTER_DEVICE_IMPL(hard_voxelize_forward_impl, MLU,
                     hard_voxelize_forward_mlu);

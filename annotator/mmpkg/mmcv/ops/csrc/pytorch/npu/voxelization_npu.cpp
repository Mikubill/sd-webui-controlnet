#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim = 3);

int hard_voxelize_forward_npu(const at::Tensor &points, at::Tensor &voxels,
                              at::Tensor &coors,
                              at::Tensor &num_points_per_voxel,
                              const std::vector<float> voxel_size,
                              const std::vector<float> coors_range,
                              const int max_points, const int max_voxels,
                              const int NDim = 3) {
  at::Tensor voxel_num_tmp = OpPreparation::ApplyTensor(points, {1});
  at::Tensor voxel_num = at_npu::native::NPUNativeFunctions::npu_dtype_cast(
      voxel_num_tmp, at::kInt);

  at::Tensor voxel_size_cpu = at::from_blob(
      const_cast<float *>(voxel_size.data()), {3}, dtype(at::kFloat));
  at::Tensor voxel_size_npu =
      CalcuOpUtil::CopyTensorHostToDevice(voxel_size_cpu);

  at::Tensor coors_range_cpu = at::from_blob(
      const_cast<float *>(coors_range.data()), {6}, dtype(at::kFloat));
  at::Tensor coors_range_npu =
      CalcuOpUtil::CopyTensorHostToDevice(coors_range_cpu);

  int64_t max_points_ = (int64_t)max_points;
  int64_t max_voxels_ = (int64_t)max_voxels;

  // only support true now
  bool deterministic = true;

  OpCommand cmd;
  cmd.Name("Voxelization")
      .Input(points)
      .Input(voxel_size_npu)
      .Input(coors_range_npu)
      .Output(voxels)
      .Output(coors)
      .Output(num_points_per_voxel)
      .Output(voxel_num)
      .Attr("max_points", max_points_)
      .Attr("max_voxels", max_voxels_)
      .Attr("deterministic", deterministic)
      .Run();
  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];
  return voxel_num_int;
}

REGISTER_NPU_IMPL(hard_voxelize_forward_impl, hard_voxelize_forward_npu);

#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void gather_points_forward_npu(int b, int c, int n, int npoints,
                               const Tensor points, const Tensor idx,
                               Tensor out) {
  // b, c, n, and npoints do not need to be passed into gatherv2,
  // b, c, n, and npoints are calculated inside the operator
  // gatherv2 operator in ascend needs to set axis to 2, batch_dims is 1
  c10::SmallVector<int64_t, N> axis = {2};
  int64_t batch_dims = 1;

  OpCommand cmd;
  cmd.Name("GatherV2")
      .Input(points)
      .Input(idx)
      .Input(axis)
      .Output(out)
      .Attr("batch_dims", batch_dims)
      .Run();
}

void gather_points_forward_impl(int b, int c, int n, int npoints,
                                const Tensor points, const Tensor idx,
                                Tensor out);

REGISTER_NPU_IMPL(gather_points_forward_impl, gather_points_forward_npu);

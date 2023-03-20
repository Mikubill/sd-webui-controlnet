#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

Tensor fused_bias_leakyrelu_op_impl(const Tensor &input, const Tensor &bias,
                                    const Tensor &refer, int act, int grad,
                                    float alpha, float scale);

Tensor fused_bias_leakyrelu_npu(const Tensor &input, const Tensor &bias,
                                const Tensor &refer, int act, int grad,
                                float alpha, float scale) {
  at::Tensor py = at::empty_like(input);
  // forward
  if (grad == 0) {
    auto input_size = input.sizes();
    int input_length = input_size.size();
    c10::SmallVector<int64_t, SIZE> input_size_tmp;
    input_size_tmp = array_to_small_vector(input_size);
    if (input_length > 1) {
      for (int i = 0; i < input_length; i++) {
        if (i != 1) {
          input_size_tmp[i] = 1;
        }
      }
    }
    at::Tensor bias_tmp = at::reshape(bias, input_size_tmp);
    at::Tensor bias_ = at_npu::native::NPUNativeFunctions::npu_broadcast(
        bias_tmp, input.sizes());
    OpCommand cmd;
    cmd.Name("FusedBiasLeakyRelu")
        .Input(input)
        .Input(bias_)
        .Output(py)
        .Attr("scale", scale)
        .Attr("negative_slope", alpha)
        .Run();
  }

  // backward
  if (grad == 1) {
    OpCommand cmd;
    cmd.Name("FusedBiasLeakyReluGrad")
        .Input(input)
        .Input(refer)
        .Output(py)
        .Attr("scale", scale)
        .Attr("negative_slope", alpha)
        .Run();
  }
  return py;
}

REGISTER_NPU_IMPL(fused_bias_leakyrelu_op_impl, fused_bias_leakyrelu_npu);

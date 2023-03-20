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

void KernelTinShiftForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *shifts, void *output, const int batch_size,
    const int time_size, const int channel_size, const int hw_size,
    const int group_size, const int group_channel,
    const cnrtDataType_t data_dtype, const int channel_per_core,
    const int max_number_hw_per_core, const int max_length_per_core);

void KernelTinShiftBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *grad_output, const void *shifts, void *grad_input,
    const int batch_size, const int time_size, const int channel_size,
    const int hw_size, const int group_size, const int group_channel,
    const cnrtDataType_t data_dtype, const int channel_per_core,
    const int max_number_hw_per_core, const int max_length_per_core);

// policy function
static void policyFunc(const Tensor &input, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, int *channel_per_core,
                       int *max_number_hw_per_core, int *max_length_per_core) {
  const int32_t cluster_limit = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  const int32_t core_limit = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto nram_size = torch_mlu::getDeviceAttr(cnrtAttrNramSizePerMcore);
  const int core_num = core_limit * cluster_limit;
  const int batch_size = input.size(0);
  const int time_size = input.size(1);
  const int channel_size = input.size(2);
  const int hw_size = input.size(3);

  const size_t size_per_channel = time_size * hw_size * input.itemsize();
  *channel_per_core = nram_size / size_per_channel;
  int task_dim = 0;
  if (*channel_per_core == 0) {
    const size_t size_per_hw = hw_size * input.itemsize();
    *max_number_hw_per_core = nram_size / size_per_hw;
    if (*max_number_hw_per_core <= 0) {
      *max_length_per_core = nram_size / input.itemsize();
    }
    int tmp_max_number_hw_per_core =
        *max_number_hw_per_core > 0 ? *max_number_hw_per_core : 1;
    const int loop_time =
        (time_size / (tmp_max_number_hw_per_core)) +
        ((time_size % (tmp_max_number_hw_per_core)) > 0 ? 1 : 0);
    task_dim = batch_size * channel_size * loop_time < core_num
                   ? batch_size * channel_size * loop_time
                   : core_num;
  } else {
    task_dim = batch_size * channel_size < core_num ? batch_size * channel_size
                                                    : core_num;
  }

  k_dim->x = core_limit;
  k_dim->y = (task_dim / core_limit) > 0 ? (task_dim / core_limit) : 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

void TINShiftForwardMLUKernelLauncher(Tensor input, Tensor shift,
                                      Tensor output) {
  // params check
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "input type should be Float or Half, got ", input.scalar_type(), ".");
  TORCH_CHECK(input.dim() == 4, "input should be a 4d tensor, got ",
              input.dim(), "d.");
  TORCH_CHECK(shift.dim() == 2, "shift should be a 2d tensor, got ",
              shift.dim(), "d.");
  TORCH_CHECK(
      input.size(0) == shift.size(0),
      "input batch size should be the same as shift's, input batch size is ",
      input.size(0), " and shift batch size is ", shift.size(0), ".");
  TORCH_CHECK(input.size(0) != 0, "Input batch size should not be zero.");
  TORCH_CHECK(input.size(3) != 0,
              "The last dim size of input should not be zero.");
  if (input.size(1) == 0) {
    return;
  }
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int channel_per_core = 0;
  int max_number_hw_per_core = 0;
  int max_length_per_core = 0;
  policyFunc(input, &k_dim, &k_type, &channel_per_core, &max_number_hw_per_core,
             &max_length_per_core);

  const int batch_size = input.size(0);
  const int time_size = input.size(1);
  const int channel_size = input.size(2);
  const int hw_size = input.size(3);
  const int group_size = shift.size(1);
  int group_channel = channel_size / group_size;

  // get tensor impl
  auto input_impl = torch_mlu::getMluTensorImpl(input);
  auto shift_impl = torch_mlu::getMluTensorImpl(shift);
  auto output_impl = torch_mlu::getMluTensorImpl(output);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get the mlu ptr
  auto input_ptr = input_impl->cnnlMalloc();
  auto shift_ptr = shift_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  cnrtDataType_t data_dtype = torch_mlu::toCnrtDtype(input.dtype());

  KernelTinShiftForward(k_dim, k_type, queue, input_ptr, shift_ptr, output_ptr,
                        batch_size, time_size, channel_size, hw_size,
                        group_size, group_channel, data_dtype, channel_per_core,
                        max_number_hw_per_core, max_length_per_core);
}

void TINShiftBackwardMLUKernelLauncher(Tensor grad_output, Tensor shift,
                                       Tensor grad_input) {
  // params check
  TORCH_CHECK(grad_output.scalar_type() == at::kFloat ||
                  grad_output.scalar_type() == at::kHalf,
              "grad_output type should be Float or Half, got ",
              grad_output.scalar_type(), ".");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output should be a 4d tensor, got ",
              grad_output.dim(), "d.");
  TORCH_CHECK(shift.dim() == 2, "shift should be a 2d tensor, got ",
              shift.dim(), "d.");
  TORCH_CHECK(grad_output.size(0) == shift.size(0),
              "grad_output batch size should be the same as shift's, "
              "grad_output batch size is ",
              grad_output.size(0), ", shift batch size is ", shift.size(0),
              ".");
  TORCH_CHECK(grad_output.size(0) != 0,
              "grad_output batch size should not be zero.");
  TORCH_CHECK(grad_output.size(3) != 0,
              "The last dim size of grad_output should not be zero.");
  if (grad_output.size(1) == 0) {
    return;
  }
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int channel_per_core = 0;
  int max_number_hw_per_core = 0;
  int max_length_per_core = 0;
  policyFunc(grad_output, &k_dim, &k_type, &channel_per_core,
             &max_number_hw_per_core, &max_length_per_core);

  const int batch_size = grad_output.size(0);
  const int time_size = grad_output.size(1);
  const int channel_size = grad_output.size(2);
  const int hw_size = grad_output.size(3);
  const int group_size = shift.size(1);
  int group_channel = channel_size / group_size;

  // get tensor impl
  auto grad_output_impl = torch_mlu::getMluTensorImpl(grad_output);
  auto shift_impl = torch_mlu::getMluTensorImpl(shift);
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get the mlu ptr
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto shift_ptr = shift_impl->cnnlMalloc();
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();

  cnrtDataType_t data_dtype = torch_mlu::toCnrtDtype(grad_output.dtype());

  KernelTinShiftBackward(k_dim, k_type, queue, grad_output_ptr, shift_ptr,
                         grad_input_ptr, batch_size, time_size, channel_size,
                         hw_size, group_size, group_channel, data_dtype,
                         channel_per_core, max_number_hw_per_core,
                         max_length_per_core);
}

void tin_shift_forward_mlu(Tensor input, Tensor shift, Tensor output) {
  TINShiftForwardMLUKernelLauncher(input, shift, output);
}

void tin_shift_backward_mlu(Tensor grad_output, Tensor shift,
                            Tensor grad_input) {
  TINShiftBackwardMLUKernelLauncher(grad_output, shift, grad_input);
}

void tin_shift_forward_impl(Tensor input, Tensor shift, Tensor output);

void tin_shift_backward_impl(Tensor grad_output, Tensor shift,
                             Tensor grad_input);

REGISTER_DEVICE_IMPL(tin_shift_forward_impl, MLU, tin_shift_forward_mlu);
REGISTER_DEVICE_IMPL(tin_shift_backward_impl, MLU, tin_shift_backward_mlu);

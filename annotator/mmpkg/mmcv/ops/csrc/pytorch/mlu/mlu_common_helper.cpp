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
#include "mlu_common_helper.h"

// Descriptors
mluOpDataType_t getMluOpDataType(const caffe2::TypeMeta& data_type) {
  const std::map<std::string, mluOpDataType_t> mapping_type = {
      {std::string("c10::Half"), MLUOP_DTYPE_HALF},
      {std::string("float"), MLUOP_DTYPE_FLOAT},
      {std::string("double"), MLUOP_DTYPE_DOUBLE},
      {std::string("int8"), MLUOP_DTYPE_INT8},
      {std::string("signed char"), MLUOP_DTYPE_INT8},
      {std::string("short int"), MLUOP_DTYPE_INT16},
      {std::string("short"), MLUOP_DTYPE_INT16},
      {std::string("int"), MLUOP_DTYPE_INT32},
      {std::string("long int"), MLUOP_DTYPE_INT64},
      {std::string("long"), MLUOP_DTYPE_INT64},
      {std::string("unsigned char"), MLUOP_DTYPE_UINT8},
      {std::string("bool"), MLUOP_DTYPE_BOOL},
      {std::string("c10::complex<c10::Half>"), MLUOP_DTYPE_COMPLEX_HALF},
      {std::string("c10::complex<float>"), MLUOP_DTYPE_COMPLEX_FLOAT}};

  if (mapping_type.find(std::string(data_type.name())) != mapping_type.end()) {
    return mapping_type.find(std::string(data_type.name()))->second;
  }
  return MLUOP_DTYPE_INVALID;
}

// laytout
mluOpTensorLayout_t getMluOpSuggestLayout(const at::Tensor& input) {
  auto suggest_memory_format = input.suggest_memory_format();
  mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY;
  switch (input.dim()) {
    case 4:
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast)
                   ? MLUOP_LAYOUT_NHWC
                   : MLUOP_LAYOUT_NCHW;
      break;
    case 5:
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast3d)
                   ? MLUOP_LAYOUT_NDHWC
                   : MLUOP_LAYOUT_NCDHW;
      break;
    default:
      layout = MLUOP_LAYOUT_ARRAY;
  }
  return layout;
}

void MluOpTensorDescriptor::set(Tensor t) {
  mluOpDataType_t data_type = getMluOpDataType(t.dtype());
  mluOpTensorLayout_t layout = getMluOpSuggestLayout(t);
  int t_dim = t.dim();
  std::vector<int> dim_array;
  if (t_dim == 0) {
    dim_array.push_back(
        1);  // ScalarTensor(0-dim 1-item Tensor) view like size = 1 as default;
  } else {
    for (int i = 0; i < t_dim; i++) {
      dim_array.push_back(static_cast<int>(t.sizes().vec()[i]));
    }
  }
  set_desc(t, layout, data_type, dim_array);
}

void MluOpTensorDescriptor::set_desc(const at::Tensor& t,
                                     mluOpTensorLayout_t layout,
                                     mluOpDataType_t dtype,
                                     std::vector<int>& dims) {
  int dimNb = dims.size();
  mluOpSetTensorDescriptor(desc_, layout, dtype, dimNb, dims.data());
}

// Handles
std::once_flag mmcv_mluop_init_flag;
std::mutex mmcv_mluop_mutex;
static std::vector<MluOpHandle> mmcv_mluop_handles;

mluOpHandle_t mluOpGetCurrentHandle(c10::DeviceIndex device_index) {
  std::call_once(mmcv_mluop_init_flag,
                 []()  // Init mmcv_mluop_handles 1-device <-> 1-handle
                 {
                   c10::DeviceIndex num_devices = torch_mlu::device_count();
                   mmcv_mluop_handles.resize(num_devices);
                 });

  if (device_index == -1) {
    device_index = torch_mlu::current_device();
  }
  std::lock_guard<std::mutex> mmcv_mluop_guard(mmcv_mluop_mutex);
  auto queue = torch_mlu::getCurrentQueue(device_index).queue();
  mmcv_mluop_handles[device_index].setQueue(queue);
  return mmcv_mluop_handles[device_index].handle;
}

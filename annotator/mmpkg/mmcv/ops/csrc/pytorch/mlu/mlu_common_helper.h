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
#pragma once
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include "aten.h"
#include "mlu_op.h"
#include "pytorch_device_registry.hpp"

#define MLUOP_MAJOR 0
#define MLUOP_MINOR 4
#define MLUOP_PATCHLEVEL 2

mluOpDataType_t getMluOpDataType(const caffe2::TypeMeta& data_type);
mluOpTensorLayout_t getMluOpSuggestLayout(const at::Tensor& input);

class MluOpTensorDescriptor {
 public:
  MluOpTensorDescriptor() { mluOpCreateTensorDescriptor(&desc_); };
  ~MluOpTensorDescriptor() { mluOpDestroyTensorDescriptor(desc_); }

  void set(at::Tensor);
  mluOpTensorDescriptor_t desc() { return desc_; }

 private:
  mluOpTensorDescriptor_t desc_;
  void set_desc(const at::Tensor&, mluOpTensorLayout_t, mluOpDataType_t,
                std::vector<int>& dims);
};

mluOpHandle_t mluOpGetCurrentHandle(c10::DeviceIndex device_index = -1);

class MluOpHandle {
 public:
  MluOpHandle() : handle(nullptr) { mluOpCreate(&handle); }
  ~MluOpHandle() {
    if (handle) {
      mluOpDestroy(handle);
      handle = nullptr;
    }
  }
  void setQueue(cnrtQueue_t queue) { mluOpSetQueue(handle, queue); }
  mluOpHandle_t handle;
};

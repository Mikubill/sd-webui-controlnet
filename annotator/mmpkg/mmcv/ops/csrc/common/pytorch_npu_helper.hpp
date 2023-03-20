/******************************************************************************
 * Copyright (c) 2022 Huawei Technologies Co., Ltd
 * All rights reserved.
 *
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef PYTORCH_NPU_HELPER_HPP_
#define PYTORCH_NPU_HELPER_HPP_

#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

#define NPU_NAME_SPACE at_npu::native

#define REGISTER_NPU_IMPL(key, value) REGISTER_DEVICE_IMPL(key, XLA, value)

#define CHECK_NPU(x) \
  TORCH_CHECK(x.device().type() == at::kXLA, #x " must be a NPU tensor")

#endif  // PYTORCH_NPU_HELPER_HPP_

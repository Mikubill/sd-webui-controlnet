// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>
#include <utils/spconv/tensorview/tensorview.h>

#include "pytorch_cuda_helper.hpp"

namespace tv {
struct GPU {
  GPU(cudaStream_t s = 0) : mStream(s) {}
  virtual cudaStream_t getStream() const { return mStream; }
  cudaStream_t mStream = 0;
};

struct TorchGPU : public tv::GPU {
  virtual cudaStream_t getStream() const override {
    return at::cuda::getCurrentCUDAStream();
  }
};

template <typename scalar_t>
void check_torch_dtype(const torch::Tensor &tensor) {
  switch (tensor.type().scalarType()) {
    case at::ScalarType::Double: {
      auto val = std::is_same<std::remove_const_t<scalar_t>, double>::value;
      TV_ASSERT_RT_ERR(val, "error");
      break;
    }
    case at::ScalarType::Float: {
      auto val = std::is_same<std::remove_const_t<scalar_t>, float>::value;
      TV_ASSERT_RT_ERR(val, "error");
      break;
    }
    case at::ScalarType::Int: {
      auto val = std::is_same<std::remove_const_t<scalar_t>, int>::value;
      TV_ASSERT_RT_ERR(val, "error");
      break;
    }
    case at::ScalarType::Half: {
      auto val = std::is_same<std::remove_const_t<scalar_t>, at::Half>::value;
      TV_ASSERT_RT_ERR(val, "error");
      break;
    }
    case at::ScalarType::Long: {
      auto val = std::is_same<std::remove_const_t<scalar_t>, long>::value;
      TV_ASSERT_RT_ERR(val, "error");
      break;
    }
    default:
      TV_ASSERT_RT_ERR(false, "error");
  }
}

template <typename scalar_t>
tv::TensorView<scalar_t> torch2tv(const torch::Tensor &tensor) {
  check_torch_dtype<scalar_t>(tensor);
  tv::Shape shape;
  for (auto i : tensor.sizes()) {
    shape.push_back(i);
  }
  return tv::TensorView<scalar_t>(
      tensor.data_ptr<std::remove_const_t<scalar_t>>(), shape);
}
}  // namespace tv

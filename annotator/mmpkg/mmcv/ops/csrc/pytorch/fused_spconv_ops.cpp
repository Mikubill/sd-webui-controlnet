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

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

torch::Tensor fused_indice_conv_batchnorm_forward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM) {
  return DISPATCH_DEVICE_IMPL(fused_indice_conv_batchnorm_forward_impl,
                              features, filters, bias, indicePairs, indiceNum,
                              numActOut, _inverse, _subM);
}

torch::Tensor fused_indice_conv_batchnorm_forward(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM) {
  return fused_indice_conv_batchnorm_forward_impl(features, filters, bias,
                                                  indicePairs, indiceNum,
                                                  numActOut, _inverse, _subM);
}

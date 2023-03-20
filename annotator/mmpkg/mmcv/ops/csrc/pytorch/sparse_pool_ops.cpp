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

torch::Tensor indice_maxpool_forward_impl(torch::Tensor features,
                                          torch::Tensor indicePairs,
                                          torch::Tensor indiceNum,
                                          int64_t numAct) {
  return DISPATCH_DEVICE_IMPL(indice_maxpool_forward_impl, features,
                              indicePairs, indiceNum, numAct);
}

torch::Tensor indice_maxpool_forward(torch::Tensor features,
                                     torch::Tensor indicePairs,
                                     torch::Tensor indiceNum, int64_t numAct) {
  return indice_maxpool_forward_impl(features, indicePairs, indiceNum, numAct);
}

torch::Tensor indice_maxpool_backward_impl(torch::Tensor features,
                                           torch::Tensor outFeatures,
                                           torch::Tensor outGrad,
                                           torch::Tensor indicePairs,
                                           torch::Tensor indiceNum) {
  return DISPATCH_DEVICE_IMPL(indice_maxpool_backward_impl, features,
                              outFeatures, outGrad, indicePairs, indiceNum);
}

torch::Tensor indice_maxpool_backward(torch::Tensor features,
                                      torch::Tensor outFeatures,
                                      torch::Tensor outGrad,
                                      torch::Tensor indicePairs,
                                      torch::Tensor indiceNum) {
  return indice_maxpool_backward_impl(features, outFeatures, outGrad,
                                      indicePairs, indiceNum);
}

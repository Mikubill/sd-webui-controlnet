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

template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsForwardCUDAKernelLauncher(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward_cuda(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  return GetIndicePairsForwardCUDAKernelLauncher<NDim>(
      indices, batchSize, outSpatialShape, spatialShape, kernelSize, stride,
      padding, dilation, outPadding, _subM, _transpose);
};

template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsBackwardCUDAKernelLauncher(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_backward_cuda(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  return GetIndicePairsBackwardCUDAKernelLauncher<NDim>(
      indices, gridOut, batchSize, outSpatialShape, spatialShape, kernelSize,
      stride, padding, dilation, outPadding, _subM, _transpose);
};

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  if (indices.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(indices);

    return get_indice_pairs_forward_cuda<NDim>(
        indices, batchSize, outSpatialShape, spatialShape, kernelSize, stride,
        padding, dilation, outPadding, _subM, _transpose);
#else
    AT_ERROR("get_indice_pairs is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("get_indice_pairs is not implemented on CPU");
  }
}

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_backward(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  if (indices.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(indices);
    CHECK_CUDA_INPUT(gridOut);

    return get_indice_pairs_backward_cuda<NDim>(
        indices, gridOut, batchSize, outSpatialShape, spatialShape, kernelSize,
        stride, padding, dilation, outPadding, _subM, _transpose);
#else
    AT_ERROR("get_indice_pairs is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("get_indice_pairs is not implemented on CPU");
  }
}

torch::Tensor indice_conv_forward_impl(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM) {
  return DISPATCH_DEVICE_IMPL(indice_conv_forward_impl, features, filters,
                              indicePairs, indiceNum, numActOut, _inverse,
                              _subM);
}

torch::Tensor indice_conv_forward(torch::Tensor features, torch::Tensor filters,
                                  torch::Tensor indicePairs,
                                  torch::Tensor indiceNum, int64_t numActOut,
                                  int64_t _inverse, int64_t _subM) {
  return indice_conv_forward_impl(features, filters, indicePairs, indiceNum,
                                  numActOut, _inverse, _subM);
}

std::vector<torch::Tensor> indice_conv_backward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  return DISPATCH_DEVICE_IMPL(indice_conv_backward_impl, features, filters,
                              outGrad, indicePairs, indiceNum, _inverse, _subM);
}

std::vector<torch::Tensor> indice_conv_backward(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  return indice_conv_backward_impl(features, filters, outGrad, indicePairs,
                                   indiceNum, _inverse, _subM);
}

template std::vector<torch::Tensor> get_indice_pairs_forward<2>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template std::vector<torch::Tensor> get_indice_pairs_forward<3>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template std::vector<torch::Tensor> get_indice_pairs_forward<4>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template std::vector<torch::Tensor> get_indice_pairs_backward<2>(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template std::vector<torch::Tensor> get_indice_pairs_backward<3>(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

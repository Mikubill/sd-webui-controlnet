// Copyright (c) OpenMMLab. All rights reserved
#include <stdio.h>

#include <vector>

#include "common_cuda_helper.hpp"
#include "trt_cuda_helper.cuh"
#include "trt_plugin_helper.hpp"

static int const threadsPerBlock = sizeof(unsigned long long int) * 8;

using mmcv::TensorDesc;

template <typename T>
__global__ void onnx_scatternd_kernel(const int n, const int* indices,
                                      const T* update, T* output,
                                      TensorDesc tensor_desc,
                                      TensorDesc indice_desc) {
  const int indice_cols = indice_desc.shape[indice_desc.dim - 1];
  const int copy_stride = tensor_desc.stride[indice_cols - 1];
  const int* stride = &(tensor_desc.stride[0]);
  CUDA_1D_KERNEL_LOOP(index, n) {
    int output_offset = 0;
    const int* indices_current = indices + index * indice_cols;
    for (int i = 0; i < indice_cols; ++i) {
      output_offset += stride[i] * indices_current[i];
    }
    memcpy(output + output_offset, update + index * copy_stride,
           copy_stride * sizeof(T));
  }
}

template <typename T>
void TRTONNXScatterNDKernelLauncher(const T* data, const int* indices,
                                    const T* update, const int* dims,
                                    int nbDims, const int* indices_dims,
                                    int indice_nbDims, T* output,
                                    cudaStream_t stream) {
  // fill tensordesc and initial
  TensorDesc tensor_desc;
  memset((void*)&tensor_desc, 0, sizeof(TensorDesc));
  tensor_desc.dim = nbDims;
  tensor_desc.shape[nbDims - 1] = dims[nbDims - 1];
  tensor_desc.stride[nbDims - 1] = 1;
  for (int i = nbDims - 2; i >= 0; --i) {
    tensor_desc.shape[i] = dims[i];
    tensor_desc.stride[i] = dims[i + 1] * tensor_desc.stride[i + 1];
  }
  const int data_size = tensor_desc.stride[0] * tensor_desc.shape[0];

  TensorDesc indice_desc;
  memset((void*)&indice_desc, 0, sizeof(TensorDesc));
  indice_desc.dim = indice_nbDims;
  indice_desc.shape[indice_nbDims - 1] = indices_dims[indice_nbDims - 1];
  indice_desc.stride[indice_nbDims - 1] = 1;
  for (int i = indice_nbDims - 2; i >= 0; --i) {
    indice_desc.shape[i] = indices_dims[i];
    indice_desc.stride[i] = indices_dims[i + 1] * indice_desc.stride[i + 1];
  }

  // output = np.copy(data)
  cudaMemcpyAsync(output, data, data_size * sizeof(T),
                  cudaMemcpyDeviceToDevice);

  int num_update_indice = 1;
  for (int i = 0; i < indice_nbDims - 1; ++i) {
    num_update_indice *= indice_desc.shape[i];
  }
  // scatter
  const int col_block = GET_BLOCKS(num_update_indice, threadsPerBlock);
  onnx_scatternd_kernel<<<col_block, threadsPerBlock, 0, stream>>>(
      num_update_indice, indices, update, output, tensor_desc, indice_desc);
}

void TRTONNXScatterNDKernelLauncher_float(const float* data, const int* indices,
                                          const float* update, const int* dims,
                                          int nbDims, const int* indices_dims,
                                          int indice_nbDims, float* output,
                                          cudaStream_t stream) {
  TRTONNXScatterNDKernelLauncher<float>(data, indices, update, dims, nbDims,
                                        indices_dims, indice_nbDims, output,
                                        stream);
}

void TRTONNXScatterNDKernelLauncher_int32(const int* data, const int* indices,
                                          const int* update, const int* dims,
                                          int nbDims, const int* indices_dims,
                                          int indice_nbDims, int* output,
                                          cudaStream_t stream) {
  TRTONNXScatterNDKernelLauncher<int>(data, indices, update, dims, nbDims,
                                      indices_dims, indice_nbDims, output,
                                      stream);
}

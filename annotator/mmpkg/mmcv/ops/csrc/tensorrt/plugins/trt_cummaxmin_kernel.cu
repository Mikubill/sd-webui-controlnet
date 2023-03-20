// Copyright (c) OpenMMLab. All rights reserved

#include "common_cuda_helper.hpp"
#include "trt_cuda_helper.cuh"
#include "trt_plugin_helper.hpp"

using mmcv::TensorDesc;

template <typename scalar_t>
__global__ void cummaxmin_kernel(const scalar_t *input, scalar_t *output_value,
                                 int *output_index, TensorDesc tensor_desc,
                                 int cum_dim, int cum_type) {
  const size_t cum_size = tensor_desc.shape[cum_dim];
  const size_t cum_stride = tensor_desc.stride[cum_dim];
  const size_t data_size =
      tensor_desc.stride[0] * tensor_desc.shape[0] / cum_size;
  CUDA_1D_KERNEL_LOOP(index, data_size) {
    size_t cum_offset =
        index / cum_stride * (cum_size * cum_stride) + index % cum_stride;
    int cum_index = 0;
    auto cum_value = input[cum_offset];
    output_value[cum_offset] = cum_value;
    output_index[cum_offset] = cum_index;

    for (size_t cum_index_current = 1; cum_index_current < cum_size;
         ++cum_index_current) {
      cum_offset += cum_stride;
      const auto cum_value_current = input[cum_offset];
      switch (cum_type) {
        case 0:  // max
          if (cum_value_current > cum_value) {
            cum_value = cum_value_current;
            cum_index = cum_index_current;
          }
          break;
        case 1:  // min
          if (cum_value_current < cum_value) {
            cum_value = cum_value_current;
            cum_index = cum_index_current;
          }
          break;
      }
      output_value[cum_offset] = cum_value;
      output_index[cum_offset] = cum_index;
    }
  }
}

template <typename scalar_t>
void CumMaxMinForwardLauncher(const scalar_t *input, scalar_t *output_value,
                              int *output_index, const int *dims, int nbDims,
                              int cum_dim, int cum_type, cudaStream_t stream) {
  // fill tensordesc and initial
  TensorDesc tensor_desc;
  memset((void *)&tensor_desc, 0, sizeof(TensorDesc));
  tensor_desc.dim = nbDims;
  tensor_desc.shape[nbDims - 1] = dims[nbDims - 1];
  tensor_desc.stride[nbDims - 1] = 1;
  for (int i = nbDims - 2; i >= 0; --i) {
    tensor_desc.shape[i] = dims[i];
    tensor_desc.stride[i] = dims[i + 1] * tensor_desc.stride[i + 1];
  }

  // cum dim should be larger than 0
  cum_dim = cum_dim >= 0 ? cum_dim : (nbDims + cum_dim);

  const int data_size =
      tensor_desc.stride[0] * tensor_desc.shape[0] / tensor_desc.shape[cum_dim];

  const int col_block = GET_BLOCKS(data_size, THREADS_PER_BLOCK);

  cummaxmin_kernel<scalar_t><<<col_block, THREADS_PER_BLOCK, 0, stream>>>(
      input, output_value, output_index, tensor_desc, cum_dim, cum_type);
}

void CumMaxMinForwardLauncher_float(const float *input, float *output_value,
                                    int *output_index, const int *dims,
                                    int nbDims, int cum_dim, int cum_type,
                                    cudaStream_t stream) {
  CumMaxMinForwardLauncher<float>(input, output_value, output_index, dims,
                                  nbDims, cum_dim, cum_type, stream);
}

void CumMaxMinForwardLauncher_int32(const int *input, int *output_value,
                                    int *output_index, const int *dims,
                                    int nbDims, int cum_dim, int cum_type,
                                    cudaStream_t stream) {
  CumMaxMinForwardLauncher<int>(input, output_value, output_index, dims, nbDims,
                                cum_dim, cum_type, stream);
}

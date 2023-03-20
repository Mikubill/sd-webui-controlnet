// Copyright (c) OpenMMLab. All rights reserved
#ifndef TRT_CUDA_HELPER_HPP
#define TRT_CUDA_HELPER_HPP
#include <cublas_v2.h>

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(0);                                                 \
    }                                                          \
  }

/**
 * Returns a view of the original tensor with its dimensions permuted.
 *
 * @param[out] dst pointer to the destination tensor
 * @param[in] src pointer to the source tensor
 * @param[in] src_size shape of the src tensor
 * @param[in] permute The desired ordering of dimensions
 * @param[in] src_dim dim of src tensor
 * @param[in] stream cuda stream handle
 */
template <class scalar_t>
void memcpyPermute(scalar_t* dst, const scalar_t* src, int* src_size,
                   int* permute, int src_dim, cudaStream_t stream = 0);

template <typename scalar_t>
cublasStatus_t cublasGemmWrap(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const scalar_t* alpha, const scalar_t* A, int lda,
                              const scalar_t* B, int ldb, const scalar_t* beta,
                              scalar_t* C, int ldc) {
  return CUBLAS_STATUS_INTERNAL_ERROR;
}

#endif  // TRT_CUDA_HELPER_HPP

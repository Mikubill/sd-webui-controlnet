#pragma once
// from pytorch.aten
#include "tensorview.h"
namespace tv {
namespace launch {

template <typename T1, typename T2>
inline int DivUp(const T1 a, const T2 b) {
  return (a + b - 1) / b;
}

constexpr int CUDA_NUM_THREADS = 1024;
inline int getBlocks(const int N) {
  TV_ASSERT_RT_ERR(N > 0,
                   "CUDA kernel launch blocks must be positive, but got N=", N);
  return DivUp(N, CUDA_NUM_THREADS);
}
}  // namespace launch
}  // namespace tv

// Copyright (c) OpenMMLab. All rights reserved
#ifndef ASSIGN_SCORE_WITHK_CUDA_KERNEL_CUH
#define ASSIGN_SCORE_WITHK_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

// input: points(B,N0,M,O), centers(B,N0,M,O), scores(B,N1,K,M), knn_idx(B,N1,K)
// output: fout(B,O,N)
// algo: fout(b,i,k,j) = s(b,i,k,m)*p(b,c(i),k,m,j) =  s(b,i,k,m)*p(b,i(k),m,j)
//       i(k) = idx(b,i,k)
//      sum: fout(b,i,j) = fout(b,i,j) + s(b,i,k,m)*p(b,i,k,m,j)
//      avg: fout(b,i,j) = sum(fout(b,i,k,j)) / k
//      max: fout(b,i,j) = max(fout(b,i,k,j), sum(s(b,i,k,m)*p(b,i,k,m,j)))

template <typename T>
__global__ void assign_score_withk_forward_cuda_kernel(
    const int B, const int N0, const int N1, const int M, const int K,
    const int O, const int aggregate, const T* points, const T* centers,
    const T* scores, const int64_t* knn_idx, T* output) {
  // ----- parallel loop for B, N1, K and O ---------
  CUDA_1D_KERNEL_LOOP(i, B * O * N1 * K) {
    // ------- loop for M ----------
    const int b = (int)(i / (O * N1 * K));
    const int o = (int)(i % (O * N1 * K) / (N1 * K));
    const int n = (int)(i % (N1 * K) / K);
    const int k = (int)(i % K);
    const int cn = (int)knn_idx[b * K * N1 + n * K +
                                0];  // The first neighbor is the center point
    const int kn = (int)knn_idx[b * K * N1 + n * K + k];
    if (kn >= N0 ||
        kn < 0) {  // if index overflows, it is out of the neighborhood range
      return;
    }
    assert(b < B);
    assert(kn < N0);
    assert(cn < N0);
    assert(o < O);
    assert(n < N1);
    const int out_idx = b * N1 * O * K + o * N1 * K + n * K + k;
    T val = output[out_idx];
    for (int m = 0; m < M; m++) {
      val += points[b * N0 * M * O + kn * M * O + m * O + o] *
                 scores[b * N1 * K * M + n * K * M + k * M + m] -
             centers[b * N0 * M * O + cn * M * O + m * O + o] *
                 scores[b * N1 * K * M + n * K * M + k * M + m];
    }
    output[out_idx] = val;
  }
}

template <typename T>
__global__ void assign_score_withk_points_backward_cuda_kernel(
    const int B, const int N0, const int N, const int M, const int K,
    const int O, const int aggregate, const T* grad_out, const T* scores,
    const int64_t* knn_idx, T* grad_points, T* grad_centers) {
  // ----- parallel loop for B, M, O ---------
  CUDA_1D_KERNEL_LOOP(i, B * M * O) {
    int b = (int)(i / (M * O));
    int m = (int)(i % (M * O) / O);
    int o = (int)(i % O);

    // ----- loop for N,K ---------
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        int kn = knn_idx[b * N * K + n * K + k];
        int cn = knn_idx[b * N * K + n * K + 0];
        if (kn >= N0 || kn < 0) {  // if index overflows, it is out of the
                                   // neighborhood range
          continue;
        }
        atomicAdd(grad_points + b * N0 * M * O + kn * M * O + m * O + o,
                  scores[b * N * K * M + n * K * M + k * M + m] *
                      grad_out[b * O * N * K + o * N * K + n * K + k]);
        atomicAdd(grad_centers + b * N0 * M * O + cn * M * O + m * O + o,
                  -scores[b * N * K * M + n * K * M + k * M + m] *
                      grad_out[b * O * N * K + o * N * K + n * K + k]);
      }
    }
  }
}

template <typename T>
__global__ void assign_score_withk_scores_backward_cuda_kernel(
    const int B, const int N0, const int N, const int M, const int K,
    const int O, const int aggregate, const T* grad_out, const T* points,
    const T* centers, const int64_t* knn_idx, T* grad_scores) {
  // ----- parallel loop for B, N, K, M ---------
  CUDA_1D_KERNEL_LOOP(i, B * N * K * M) {
    const int b = (int)(i / (N * M * K));
    const int n = (int)(i % (N * M * K) / M / K);
    const int k = (int)(i % (M * K) / M);
    const int m = (int)(i % M);
    const int cn = knn_idx[b * N * K + n * K + 0];
    const int kn = knn_idx[b * N * K + n * K + k];
    if (kn >= N0 ||
        kn < 0) {  // if index overflows, it is out of the neighborhood range
      return;
    }

    // -------------- loop for O ------------------------
    const int out_idx = b * N * K * M + n * K * M + k * M + m;
    T val = grad_scores[out_idx];
    for (int o = 0; o < O; o++) {
      val += (points[b * N0 * M * O + kn * M * O + m * O + o] -
              centers[b * N0 * M * O + cn * M * O + m * O + o]) *
             grad_out[b * O * N * K + o * N * K + n * K + k];
    }
    grad_scores[out_idx] = val;
  }
}

#endif  // ASSIGN_SCORE_WITHK_CUDA_KERNEL_CUH

// Copyright (c) OpenMMLab. All rights reserved
#ifndef ASSIGN_SCORE_WITHK_PYTORCH_H
#define ASSIGN_SCORE_WITHK_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void assign_score_withk_forward(const Tensor& points, const Tensor& centers,
                                const Tensor& scores, const Tensor& knn_idx,
                                Tensor& output, int B, int N0, int N1, int M,
                                int K, int O, int aggregate);

void assign_score_withk_backward(const Tensor& grad_out, const Tensor& points,
                                 const Tensor& centers, const Tensor& scores,
                                 const Tensor& knn_idx, Tensor& grad_points,
                                 Tensor& grad_centers, Tensor& grad_scores,
                                 int B, int N0, int N1, int M, int K, int O,
                                 int aggregate);

#endif  // ASSIGN_SCORE_WITHK_PYTORCH_H
